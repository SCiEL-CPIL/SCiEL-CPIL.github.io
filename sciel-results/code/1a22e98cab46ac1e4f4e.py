"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Define 10 fixed centers for Radial Basis Functions
    # 3x3 grid covering [-1, 1] x [-1, 1]
    centers = np.array([
        [-1.0, -1.0],
        [-1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0, -1.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  0.0],
        [ 1.0,  1.0],
        [ 0.5,  0.5]  # 10th center
    ])
    
    # Width parameter for RBFs
    # gamma = 1 / (2 * sigma^2)
    # sigma = 0.5 -> gamma = 2.0
    gamma = 2.0
    
    # Calculate distances squared from each center
    # x and y are arrays of shape (N,)
    # centers is (10, 2)
    
    # Efficient calculation:
    # x_diff = x[:, None] - centers[None, :, 0]  # (N, 10)
    # y_diff = y[:, None] - centers[None, :, 1]  # (N, 10)
    # dist_sq = x_diff**2 + y_diff**2            # (N, 10)
    # basis = np.exp(-gamma * dist_sq)           # (N, 10)
    # pred = basis @ p                           # (N,)
    
    # Ensure input shapes are compatible (flatten if needed, though evaluator usually passes 1D)
    x_flat = np.asarray(x).flatten()
    y_flat = np.asarray(y).flatten()
    
    # If x is already 1D, this is cheap.
    # Let's do it explicitly for clarity and safety
    x_col = x_flat[:, np.newaxis]
    y_col = y_flat[:, np.newaxis]
    
    cx = centers[:, 0][np.newaxis, :]
    cy = centers[:, 1][np.newaxis, :]
    
    dx = x_col - cx
    dy = y_col - cy
    
    dist_sq = dx**2 + dy**2
    
    # Compute basis functions
    basis = np.exp(-gamma * dist_sq)
    
    # Linear combination
    pred = basis @ p
    
    return pred
# EVOLVE-BLOCK-END
