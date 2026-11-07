"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF approach: Sum of Gaussian kernels
    # 10 parameters correspond to 10 Gaussian centers
    # Centers are fixed, evenly spaced across the domain [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    # Fixed width for all Gaussians
    sigma = 1.0
    
    # x shape: (N,)
    # centers shape: (10,)
    # We want to compute exp(-0.5 * ((x_i - c_j) / sigma)^2) for all i, j
    # x[:, None] has shape (N, 1)
    # centers[None, :] has shape (1, 10)
    # diff has shape (N, 10)
    diff = (x[:, None] - centers[None, :]) / sigma
    
    # Compute the basis matrix
    # basis[i, j] = exp(-0.5 * diff[i, j]^2)
    basis = np.exp(-0.5 * diff**2)
    
    # prediction = basis @ params
    # basis shape: (N, 10)
    # params shape: (10,)
    return basis @ params
# EVOLVE-BLOCK-END
