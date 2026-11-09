"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define fixed centers for the Gaussian RBFs
    centers = np.linspace(-3.0, 3.0, 10)
    width = 1.0
    
    # Compute the basis matrix
    # x is shape (N,), centers is shape (10,)
    # We need a matrix of shape (N, 10) where M[i, j] = exp(-0.5 * ((x[i] - centers[j]) / width)^2)
    
    # Reshape x to (N, 1) and centers to (1, 10) for broadcasting
    x_col = x.reshape(-1, 1)
    centers_row = centers.reshape(1, -1)
    
    diffs = x_col - centers_row
    basis = np.exp(-0.5 * (diffs / width) ** 2)
    
    # Linear combination
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
