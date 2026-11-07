"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) acquisition function
    # EI(x) = E[max(f_best - f(x), 0)]
    # For a Gaussian posterior f(x) ~ N(mu, sigma^2):
    # If sigma > 0:
    #   EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    #   where z = (f_best - mu) / sigma
    # If sigma == 0:
    #   EI = max(f_best - mu, 0)
    
    # Handle numerical stability and edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
