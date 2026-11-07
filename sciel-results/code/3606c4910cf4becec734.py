"""Rewritten program for symbolic regression with RBF kernel approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a radial basis function (RBF) kernel approximation.
    
    The basis consists of 10 RBF terms centered at strategic points in the input space [0.5, 2]^3.
    Each RBF is a Gaussian kernel: exp(-||p - c_i||^2 / (2*sigma^2))
    
    Centers are chosen to cover the input domain:
    - Corner points of the cube
    - Center points along each axis
    - The global center
    
    This allows capturing smooth, non-polynomial structures that a global polynomial cannot fit.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if x.size == 0:
        return np.array([], dtype=np.float64)
    
    # Define 10 centers in the input space [0.5, 2]^3
    # Strategy: cover corners, face centers, and body center
    centers = np.array([
        [0.5, 0.5, 0.5],   # corner
        [2.0, 0.5, 0.5],   # corner
        [0.5, 2.0, 0.5],   # corner
        [2.0, 2.0, 0.5],   # corner
        [0.5, 0.5, 2.0],   # corner
        [2.0, 0.5, 2.0],   # corner
        [0.5, 2.0, 2.0],   # corner
        [2.0, 2.0, 2.0],   # corner
        [1.25, 1.25, 1.25],# center
        [1.25, 1.25, 0.5], # face center (bottom)
    ])
    
    # Number of samples
    n = x.size
    
    # Compute RBF values for each sample and each center
    # sigma controls the width of the RBF; larger sigma = smoother
    sigma = 0.8
    
    # Reshape inputs to (n, 3)
    inputs = np.column_stack((x, y, z))  # (n, 3)
    
    # Compute squared distances from each input to each center
    # inputs: (n, 3), centers: (10, 3)
    # diff: (n, 10, 3)
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, 10, 3)
    
    # Squared distances
    sq_dist = np.sum(diff ** 2, axis=2)  # (n, 10)
    
    # RBF values: exp(-sq_dist / (2 * sigma^2))
    rbf_values = np.exp(-sq_dist / (2.0 * sigma ** 2))  # (n, 10)
    
    # Compute prediction: rbf_values (n, 10) dot params (10,) -> (n,)
    pred = rbf_values @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
