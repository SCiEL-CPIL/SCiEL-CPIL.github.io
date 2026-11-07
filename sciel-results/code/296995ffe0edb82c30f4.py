"""Program for symbolic regression task using a flexible basis model."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 5 fixed centers, each with amplitude and width parameters.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            params[0:5] = Amplitudes for 5 Gaussian kernels
            params[5:10] = Widths (sigmas) for 5 Gaussian kernels
    
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for non-finite params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # 5 evenly spaced centers across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    amplitudes = p[0:5]
    widths = p[5:10]
    
    # Clip widths to avoid division by zero or extremely small values
    widths = np.clip(widths, 0.1, 10.0)
    
    # Compute RBF values: exp(-(x - center)^2 / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    sigma_sq_2 = 2.0 * widths[np.newaxis, :]**2
    rbf_values = np.exp(-(diff**2) / sigma_sq_2)
    
    # Weighted sum of RBFs
    result = rbf_values @ amplitudes
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result

# EVOLVE-BLOCK-END
