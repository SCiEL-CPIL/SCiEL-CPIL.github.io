"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with a small exploration bonus.
    # The bonus encourages slight exploration to avoid premature convergence.
    # We use a small coefficient (0.01) to keep the behavior close to standard EI.
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add small exploration bonus
    score = ei + 0.01 * sigma_safe
    
    return score
# EVOLVE-BLOCK-END
