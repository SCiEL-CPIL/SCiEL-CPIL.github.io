"""RBF-based symbolic regression model with adaptive widths."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with 5 fixed centers and adaptive widths.
    
    params[0:5]   -> weights for each RBF
    params[5:10]  -> log-widths (log_sigma) for each RBF, exp-transformed
    
    Centers are fixed at evenly spaced points in [-3, 3].
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # 5 fixed centers evenly spaced in [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    weights = p[0:5]
    # Use exp on the width parameters to ensure positive widths
    # log_sigma = params[5:10]
    # sigma = exp(log_sigma)
    # Clip log_sigma to prevent overflow in exp and ensure stability
    log_sigma = np.clip(p[5:10], -10, 10)
    sigma = np.exp(log_sigma)
    
    # Compute RBF values: exp(-0.5 * ((x - center) / sigma)^2)
    # x has shape (n,), centers has shape (5,)
    # diff has shape (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Scale by inverse width (divide by sigma)
    # sigma has shape (5,), need to broadcast to (1, 5)
    scaled_diff = diff / sigma[np.newaxis, :]
    
    # Gaussian RBF
    rbf_values = np.exp(-0.5 * scaled_diff ** 2)
    
    # Weighted sum
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
