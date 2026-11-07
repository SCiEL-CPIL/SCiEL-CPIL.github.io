"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization.
    # This approach maximizes the expected improvement over the current best value f_best.
    # It is more adaptive than UCB as it directly considers the gap to the current best.
    
    # Clip sigma to avoid numerical issues with division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate the z-score
    z = (f_best - mu) / sigma
    
    # Calculate PDF and CDF of the standard normal distribution
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement formula for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure the output is finite and non-negative
    # EI should theoretically be non-negative, but numerical errors might cause tiny negatives
    ei = np.maximum(ei, 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
