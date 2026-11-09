"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected improvement (EI) for minimization.
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We use a small epsilon to avoid division by zero.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle cases where sigma is very small (near certain improvement)
    # If sigma is tiny and mu < f_best, EI should be close to f_best - mu
    # If sigma is tiny and mu >= f_best, EI should be close to 0
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
