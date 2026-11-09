"""Symbolic regression with RBF network and fixed centers."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 10 fixed Gaussian centers spanning [0.1, 6].
    params[0]: log-width (sigma = exp(params[0]))
    params[1:11]: weights for each of the 10 RBF centers.
    """
    # Fixed centers spanning the input domain [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Extract parameters
    # Use exp to ensure positive sigma, with clipping to avoid overflow
    log_sigma = np.clip(params[0], -10, 10)
    sigma = np.exp(log_sigma)
    
    # Compute RBF activations: Gaussian kernels
    # x has shape (n,), centers has shape (10,)
    # diff has shape (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # Gaussian: exp(-0.5 * (diff / sigma)^2)
    activations = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Weighted sum: params[1:11] has shape (10,)
    weights = params[1:11]
    
    return activations @ weights
# EVOLVE-BLOCK-END
