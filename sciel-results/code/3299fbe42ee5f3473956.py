"""Rewritten symbolic regression candidate: localized RBF with fixed grid centers."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form RBF approximation with 10 free parameters.
    params: length-10 array. First 9 are RBF weights for a 3x3 grid of Gaussian centers.
            params[9] is a global linear/constant term to capture low-frequency trends.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Ensure 1D
    x = x.ravel()
    y = y.ravel()
    n = x.shape[0]
    
    # 3x3 grid of centers in [-1, 1]
    centers = np.array([
        [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
        [-1.0,  0.0], [0.0,  0.0], [1.0,  0.0],
        [-1.0,  1.0], [0.0,  1.0], [1.0,  1.0]
    ])
    
    # Compute distances from each point to each center
    # x[:, None, :] - centers[None, :, :] -> (n, 9, 2)
    dx = x[:, None, 0] - centers[None, :, 0]  # (n, 9)
    dy = y[:, None, 0] - centers[None, :, 1]  # (n, 9)
    
    # Squared distances
    dist_sq = dx**2 + dy**2  # (n, 9)
    
    # Gaussian kernels with fixed width
    # Width chosen so that kernels have significant overlap but are local enough
    sigma = 0.5
    kernels = np.exp(-dist_sq / (2.0 * sigma**2))  # (n, 9)
    
    # RBF part: sum of weights * kernels
    rbf_weights = params[:9]  # (9,)
    rbf_pred = kernels @ rbf_weights  # (n,)
    
    # Global linear/constant term to handle low-frequency components
    # This acts as a bias/linear term that complements the local RBFs
    global_term = params[9] * (1.0 + x + y)  # Simple global basis
    
    # Combine
    pred = rbf_pred + global_term
    
    return pred
# EVOLVE-BLOCK-END
