"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Radial Basis Function (RBF) network.
    The model is a linear combination of 10 Gaussian kernels centered at fixed points.
    This approach is more flexible than a low-degree polynomial for non-polynomial targets.
    """
    # Define fixed centers for the RBFs
    # 3x3 grid covering [-1, 1] x [-1, 1] plus one extra point
    centers = np.array([
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0,  1.0],
        [ 1.0,  1.0],
        [ 0.5,  0.5]  # Extra point to improve resolution in the center/right-top
    ])
    
    # Gaussian width parameter (inverse of sigma)
    # Tuned for the domain [-1, 1]
    gamma = 2.0 
    
    # Compute distances from each input point to each center
    # x, y are arrays of shape (N,)
    # centers is (10, 2)
    # We want dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    
    # Efficient computation:
    # x[:, None] - centers[None, :, 0]
    dx = x[:, None] - centers[None, :, 0]
    dy = y[:, None] - centers[None, :, 1]
    
    dist_sq = dx**2 + dy**2
    
    # Compute RBF values: exp(-gamma * dist_sq)
    rbf_values = np.exp(-gamma * dist_sq)
    
    # Linear combination with params
    # rbf_values is (N, 10)
    # params is (10,)
    # pred is (N,)
    pred = rbf_values @ params
    
    return pred
# EVOLVE-BLOCK-END
