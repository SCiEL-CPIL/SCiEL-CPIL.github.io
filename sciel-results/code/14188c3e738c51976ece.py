"""Symbolic regression with 3 inputs – improved piecewise-linear RBF approach."""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed grid of 8 corner centers spanning [0.5, 2.0] in each dimension
    centers = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 2.0],
        [0.5, 2.0, 0.5],
        [0.5, 2.0, 2.0],
        [2.0, 0.5, 0.5],
        [2.0, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 2.0, 2.0],
    ])

    # Gaussian RBFs with a moderate width; sigma chosen so that basis functions
    # have significant support across the domain but remain localised enough
    # for BFGS to attribute weight to the nearest corners.
    sigma = 0.7

    # Build point matrix (N, 3)
    points = np.stack([x, y, z], axis=1)

    # Squared distances (N, 8)
    diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)

    # RBF values (N, 8)
    rbf_vals = np.exp(-dist_sq / (2.0 * sigma * sigma))

    # params[0:8]  -> RBF amplitudes
    # params[8]    -> linear combination coefficient for (x+y+z) (single scalar to save a parameter)
    # params[9]    -> bias
    p_rbf = params[0:8]
    p_lin = params[8]
    p_bias = params[9]

    pred_rbf = rbf_vals @ p_rbf
    pred_lin = p_lin * (x + y + z)
    pred_bias = p_bias

    return pred_rbf + pred_lin + pred_bias
