# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a Radial Basis Function (RBF) network
    with 5 fixed centers and Gaussian kernels.
    
    Parameters (length 10):
    - params[0]: Global offset (bias)
    - params[1]: Width scaling factor (inverse of sigma)
    - params[2:7]: Weights for the 5 RBF centers
    - params[7:10]: Unused (set to 0 to avoid affecting optimization)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    # Fixed centers equally spaced in [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float64)
    
    # Extract parameters
    offset = params[0]
    width_scale = params[1]
    weights = params[2:7]
    
    # Ensure width_scale is positive and not too small to avoid numerical issues
    # If width_scale is 0 or negative, use a default
    if not np.isfinite(width_scale) or abs(width_scale) < 1e-6:
        width_scale = 1.0
    
    # Compute RBF values
    # x is (n,), centers is (5,)
    # diff is (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-(diff^2) / (2 * sigma^2))
    # Here sigma = 1 / width_scale, so 1/sigma^2 = width_scale^2
    # exp(-(diff^2) * width_scale^2 / 2)
    sigma_sq = 1.0 / (width_scale * width_scale)
    kernel = np.exp(-(diff * diff) / (2.0 * sigma_sq))
    
    # Combine: y = offset + sum(weights_i * kernel_i)
    # kernel is (n, 5), weights is (5,)
    rbf_output = kernel @ weights
    
    return offset + rbf_output
# EVOLVE-BLOCK-END
