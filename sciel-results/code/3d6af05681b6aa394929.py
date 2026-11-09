"""RBF-based symbolic regression model with adaptive widths."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with 5 fixed centers and adaptive widths.
    
    params[0:5]   -> weights for each RBF
    params[5:10]  -> inverse widths (sigma_inv) for each RBF, softplus-transformed
    
    Centers are fixed at evenly spaced points in [-3, 3].
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # 5 fixed centers evenly spaced in [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    weights = p[0:5]
    # Use softplus on the width parameters to ensure positive widths
    # sigma_inv = softplus(params[5:10]) = log(1 + exp(params[5:10]))
    raw_widths = p[5:10]
    # Softplus for numerical stability
    sigma_inv = np.log1p(np.exp(np.clip(raw_widths, -50, 50)))
    
    # Ensure a minimum width to avoid singularity
    sigma_inv = np.maximum(sigma_inv, 0.1)
    
    # Compute RBF values: exp(-0.5 * (sigma_inv * (x - center))^2)
    # x has shape (n,), centers has shape (5,)
    # diff has shape (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Scale by inverse width
    scaled_diff = sigma_inv[np.newaxis, :] * diff
    
    # Gaussian RBF
    rbf_values = np.exp(-0.5 * scaled_diff ** 2)
    
    # Weighted sum
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
