"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 4 kernels, learnable global width, and bias.
    
    params[0:4]  : centers of the 4 Gaussian kernels
    params[4:8]  : amplitudes of the 4 Gaussian kernels
    params[8]    : bias term
    params[9]    : global width (sigma) for all kernels
    
    The prediction is: bias + sum(amplitude_i * exp(-(x - center_i)^2 / (2 * sigma^2)))
    """
    centers = params[0:4]
    amplitudes = params[4:8]
    bias = params[8]
    sigma = params[9]
    
    # Avoid division by zero or extremely small sigma causing numerical issues
    # If sigma is very small, the RBF becomes a spike. 
    # We clamp sigma to a minimum effective value of 0.1.
    sigma_safe = np.abs(sigma) + 0.1
    
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    
    diff = x_col - c_row
    sq_dist = diff * diff
    
    # exp(-sq_dist / (2 * sigma^2))
    kernels = np.exp(-sq_dist / (2.0 * sigma_safe * sigma_safe))
    
    prediction = (kernels @ amplitudes) + bias
    
    return prediction
# EVOLVE-BLOCK-END
