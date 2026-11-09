"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 10 parameters interpreted as 5 2D centers.
    params[0:5] are x-coordinates, params[5:10] are y-coordinates.
    Uses fixed weights (1.0) and subtracts the value at the origin to allow
    for negative predictions and better centering.
    """
    cx = params[:5]
    cy = params[5:]
    
    # x, y shape (N,)
    # cx, cy shape (5,)
    
    # Compute distances to 5 centers
    # dx shape (N, 5)
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dy = y[:, np.newaxis] - cy[np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    
    gamma = 2.0
    
    # RBF values shape (N, 5)
    rbf_values = np.exp(-gamma * dist_sq)
    
    # Average to keep scale ~1
    pred = np.mean(rbf_values, axis=1)
    
    # Subtract value at origin (0,0) to allow negative values
    # dist_sq_origin shape (5,)
    dist_sq_origin = cx**2 + cy**2
    rbf_origin = np.exp(-gamma * dist_sq_origin)
    offset = np.mean(rbf_origin)
    
    pred = pred - offset
    
    return pred
# EVOLVE-BLOCK-END
