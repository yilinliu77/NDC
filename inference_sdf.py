from pathlib import Path
import torch
from tqdm import tqdm
import trimesh
import sys
import model
import cutils

import numpy as np

def get_sdf_network(device):
    # CNN_3d = model.CNN_3d_rec7
    CNN_3d = model.CNN_3d_rec7_resnet

    network_float = CNN_3d(out_bool=False, out_float=True)
    network_float = network_float.to(device)
    # network_float.load_state_dict(torch.load("weights/weights_ndc_sdf_float.pth", weights_only=True))
    network_float.load_state_dict(torch.load("weights/weights_ndcx_sdf_float.pth", weights_only=True))
    network_float.eval()

    return None, network_float

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_sdf.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"Model path {model_path} does not exist.")
        sys.exit(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network_bool, network_float = get_sdf_network(device)

    sdf = np.load(model_path).astype(np.float32)
    sdf = sdf[::2, ::2, ::2]
    sdf = (sdf * 64)
    input_sdf = torch.from_numpy(sdf).to(device)

    patch_size = 128
    num_batches = 16
    assert input_sdf.shape[0] % patch_size == 0
    num_patches = input_sdf.shape[0] // patch_size
    input_sdf = input_sdf.reshape((num_patches, patch_size, num_patches, patch_size, num_patches, patch_size))
    input_sdf = input_sdf.permute(0, 2, 4, 1, 3, 5).reshape((-1, patch_size, patch_size, patch_size))

    receptive_padding = 3  # for grid input
    padded_sdf = torch.nn.functional.pad(input_sdf, (receptive_padding, receptive_padding, receptive_padding, receptive_padding, receptive_padding, receptive_padding), mode='constant', value=10)
    padded_sdf = torch.clamp(padded_sdf, -2, 2)

    chunked_sdf = padded_sdf.chunk(num_batches, dim=0)
    pred_output_bool = []
    pred_output_float = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            for i in tqdm(range(len(chunked_sdf))):
                output_float = network_float(chunked_sdf[i][:,None]).cpu().detach().numpy()
                # output_bool = network_bool(chunked_sdf[i][:,None]).cpu().detach().numpy()
                output_bool = (chunked_sdf[i][:, None, receptive_padding:-receptive_padding, receptive_padding:-receptive_padding, receptive_padding:-receptive_padding] < 0).detach().cpu().numpy().astype(np.int32)
                pred_output_float.append(output_float)
                pred_output_bool.append(output_bool)

    pred_output_bool = np.concatenate(pred_output_bool, axis=0)
    pred_output_float = np.concatenate(pred_output_float, axis=0)
    pred_output_float = np.clip(pred_output_float, 0, 1)
    pred_output_float = pred_output_float.reshape(num_patches, num_patches, num_patches,-1, patch_size, patch_size, patch_size)
    pred_output_float = np.transpose(pred_output_float, [0, 4, 1, 5, 2, 6, 3])
    pred_output_float = pred_output_float.reshape((num_patches*patch_size, num_patches*patch_size, num_patches*patch_size, -1))
    pred_output_bool = pred_output_bool.reshape(num_patches, num_patches, num_patches,-1, patch_size, patch_size, patch_size)
    pred_output_bool = np.transpose(pred_output_bool, [0, 4, 1, 5, 2, 6, 3])
    pred_output_bool = pred_output_bool.reshape((num_patches*patch_size, num_patches*patch_size, num_patches*patch_size, -1))
    # pred_output_bool = pred_output_bool>0.5
    vertices, triangles = cutils.dual_contouring_ndc(
        np.ascontiguousarray(pred_output_bool, np.int32),
        np.ascontiguousarray(pred_output_float, np.float32),
    )
    vertices = vertices / 128 * 2 - 1
    trimesh.Trimesh(vertices, triangles).wa
    trimesh.Trimesh(vertices, triangles).export("1.ply")
