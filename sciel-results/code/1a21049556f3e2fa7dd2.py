"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement with small exploitation parameter
    # xi controls exploitation: larger xi = more exploitation
    xi = 0.001
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized improvement
    z = (f_best - mu - xi) / sigma_safe
    
    # EI formula: (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    # For minimization, improvement is f_best - mu
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip to ensure finite values
    ei = np.clip(ei, -1e10, 1e10)
    
    return ei
# EVOLVE-BLOCK-END
