"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Gaussian RBF network with 10 fixed centers on a grid
    # Each param is the weight for one RBF center
    # Fixed width for all RBFs to keep optimization well-conditioned
    # Centers arranged in a 3x3 grid plus one at origin for 10 total
    
    # Define 10 center locations in [-1, 1]^2
    centers_x = np.array([
        -0.75,  0.0,  0.75,
        -0.5,   0.5,  0.0,
        -0.25, -0.25,  0.25,
         0.25
    ])
    centers_y = np.array([
        -0.75,  0.0,  0.75,
        -0.5,   0.5,  0.0,
        -0.25, -0.25,  0.25,
         0.0
    ])
    
    # Fixed RBF width (sigma) - chosen to cover the domain well
    sigma = 0.5
    
    # Compute RBF values: sum of weights * exp(-||x - center||^2 / (2*sigma^2))
    # x, y are arrays of shape (n,)
    # Compute squared distances to each center
    # For each center i: (x - cx_i)^2 + (y - cy_i)^2
    dx = x[:, np.newaxis] - centers_x[np.newaxis, :]  # (n, 10)
    dy = y[:, np.newaxis] - centers_y[np.newaxis, :]  # (n, 10)
    dist_sq = dx ** 2 + dy ** 2  # (n, 10)
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_values = np.exp(-dist_sq / (2.0 * sigma ** 2))  # (n, 10)
    
    # Weighted sum: each param[i] multiplies rbf_values[:, i]
    # Return shape (n,)
    return rbf_values @ params
# EVOLVE-BLOCK-END
