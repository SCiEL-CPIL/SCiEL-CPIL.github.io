"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define fixed centers for Gaussian RBFs
    # 10 centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    # Fixed width for the Gaussians
    sigma = 1.0
    
    # Compute the RBF basis
    # x is shape (N,), centers is shape (10,)
    # We want basis[i, j] = exp(-(x[i] - centers[j])^2 / (2 * sigma^2))
    
    # Reshape for broadcasting
    x_col = x[:, np.newaxis]
    centers_row = centers[np.newaxis, :]
    
    # Compute squared distances
    diff = x_col - centers_row
    sq_diff = diff * diff
    
    # Compute Gaussian values
    basis = np.exp(-sq_diff / (2.0 * sigma * sigma))
    
    # Linear combination with params
    # basis is (N, 10), params is (10,)
    return basis @ params
# EVOLVE-BLOCK-END
