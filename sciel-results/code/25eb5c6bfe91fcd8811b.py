"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Localized RBF network with Gaussian kernels on a 3x3 grid of centers.
    10 params: 9 RBF weights + 1 global bias.
    Centers are evenly spaced in [-1, 1]^2.
    Gaussian width sigma is fixed to provide good local coverage.
    """
    p = params
    # 3x3 grid of centers
    cx = np.array([-1.0, 0.0, 1.0])
    cy = np.array([-1.0, 0.0, 1.0])
    
    # Compute squared distances from each point to each center
    # x and y are 1D arrays of length N
    # We want to compute for each point (x[i], y[i]) the distance to each center (cx[j], cy[k])
    
    # x[:, None, None] has shape (N, 1, 1)
    # cx[None, :, None] has shape (1, 3, 1)
    # cy[None, None, :] has shape (1, 1, 3)
    
    dx = x[:, None, None] - cx[None, :, None]  # shape (N, 3, 1)
    dy = y[:, None, None] - cy[None, None, :]  # shape (N, 1, 3)
    
    # Squared distance: (N, 3, 3)
    dist_sq = dx**2 + dy**2
    
    # Gaussian kernel with fixed width
    sigma = 0.5
    kernels = np.exp(-dist_sq / (2.0 * sigma**2))  # shape (N, 3, 3)
    
    # Reshape kernels to (N, 9)
    kernels_flat = kernels.reshape(x.shape[0], 9)
    
    # Compute prediction: sum of weights * kernels + bias
    # p[0:9] are the RBF weights, p[9] is the bias
    prediction = kernels_flat @ p[0:9] + p[9]
    
    return prediction
# EVOLVE-BLOCK-END
