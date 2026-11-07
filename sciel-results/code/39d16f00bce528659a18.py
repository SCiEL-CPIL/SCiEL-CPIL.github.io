"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Define 10 fixed centers for Radial Basis Functions
    # 3x3 grid covering [-1, 1] x [-1, 1] plus one extra
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
        [ 0.5,  0.5]
    ])
    
    # Width parameter for RBFs
    # Using a wider width (smaller gamma) for better global coverage
    # gamma = 1 / (2 * sigma^2). sigma = 0.8 -> gamma ~ 0.78
    # Let's try gamma = 1.0 for a balance
    gamma = 1.0
    
    # Calculate distances squared from each center
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    
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
