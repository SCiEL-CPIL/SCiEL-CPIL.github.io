"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF interpolant for E. coli growth rate.
    Uses 10 fixed Gaussian centers in the normalized input space.
    The 10 params are the weights for each basis function.
    """
    # Normalize inputs to [0, 1] range based on training data ranges
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Stack normalized inputs into a 2D array of shape (n, 4)
    inputs = np.column_stack([b_norm, s_norm, temp_norm, ph_norm])
    
    # Define 10 fixed centers: 4 corners + 4 mid-edges + 1 centroid + 1 extra
    centers = np.array([
        [0.0, 0.0, 0.0, 0.0],   # corner 1
        [1.0, 1.0, 1.0, 1.0],   # corner 2
        [0.0, 1.0, 0.0, 1.0],   # corner 3
        [1.0, 0.0, 1.0, 0.0],   # corner 4
        [0.5, 0.5, 0.5, 0.5],   # centroid
        [0.5, 0.0, 0.5, 0.5],   # mid-edge 1
        [0.5, 1.0, 0.5, 0.5],   # mid-edge 2
        [0.0, 0.5, 0.5, 0.5],   # mid-edge 3
        [1.0, 0.5, 0.5, 0.5],   # mid-edge 4
        [0.5, 0.5, 0.0, 0.5],   # mid-edge 5
    ])
    
    # Fixed Gaussian width (inverse bandwidth)
    sigma = 0.4
    
    # Compute squared distances from each input point to each center
    # inputs: (n, 4), centers: (10, 4)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, 10, 4)
    sq_dist = np.sum(diff ** 2, axis=2)  # (n, 10)
    
    # Gaussian basis functions: exp(-sq_dist / (2 * sigma^2))
    basis = np.exp(-sq_dist / (2.0 * sigma ** 2))  # (n, 10)
    
    # Weighted sum of basis functions
    mu = basis @ params  # (n,)
    
    return mu
# EVOLVE-BLOCK-END
