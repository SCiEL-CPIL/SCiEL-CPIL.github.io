"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define 10 fixed RBF centers and their corresponding betas (inverse widths)
    # Strategy: Multi-scale basis.
    # 5 Wide RBFs for global trends (corners + center)
    # 5 Narrow RBFs for local details (center + cross)
    
    centers = np.array([
        [-1.0, -1.0],  # 0: Wide, Corner
        [ 1.0, -1.0],  # 1: Wide, Corner
        [-1.0,  1.0],  # 2: Wide, Corner
        [ 1.0,  1.0],  # 3: Wide, Corner
        [ 0.0,  0.0],  # 4: Wide, Center
        [ 0.0,  0.0],  # 5: Narrow, Center
        [ 0.5,  0.0],  # 6: Narrow, Right
        [-0.5,  0.0],  # 7: Narrow, Left
        [ 0.0,  0.5],  # 8: Narrow, Top
        [ 0.0, -0.5],  # 9: Narrow, Bottom
    ])
    
    betas = np.array([
        0.5,  # Wide
        0.5,  # Wide
        0.5,  # Wide
        0.5,  # Wide
        0.5,  # Wide
        4.0,  # Narrow
        4.0,  # Narrow
        4.0,  # Narrow
        4.0,  # Narrow
        4.0,  # Narrow
    ])
    
    x_col = x[:, np.newaxis]  # (N, 1)
    y_col = y[:, np.newaxis]  # (N, 1)
    
    cx = centers[:, 0]  # (10,)
    cy = centers[:, 1]  # (10,)
    
    dx = x_col - cx[np.newaxis, :]  # (N, 10)
    dy = y_col - cy[np.newaxis, :]  # (N, 10)
    dist_sq = dx**2 + dy**2  # (N, 10)
    
    # Compute RBF values: exp(-beta_j * dist_ij^2)
    # betas shape (10,), dist_sq shape (N, 10)
    rbf_values = np.exp(-betas[np.newaxis, :] * dist_sq)
    
    pred = rbf_values @ params
    
    return pred
# EVOLVE-BLOCK-END
