"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 9 Gaussian kernels and a learnable width.
    params[0] = log-width, params[1:10] = RBF coefficients.
    """
    # 9 fixed centers evenly spaced in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 9)
    
    # Width from log-param to ensure positivity
    log_width = params[0]
    # Clamp log_width to prevent overflow/underflow
    log_width = np.clip(log_width, -10.0, 10.0)
    width = np.exp(log_width)
    
    # Compute Gaussian kernels: exp(-width * (x - c_i)^2)
    # x shape: (N,), centers shape: (9,)
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    kernels = np.exp(-width * diff ** 2)
    
    # coeffs shape: (9,)
    coeffs = params[1:10]
    
    # Result: sum over i of coeffs[i] * kernels[:, i]
    result = kernels @ coeffs
    
    return result
# EVOLVE-BLOCK-END
