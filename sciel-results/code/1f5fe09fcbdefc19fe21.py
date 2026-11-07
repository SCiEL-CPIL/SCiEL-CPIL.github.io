"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) for minimization.
    
    PI is a probabilistic acquisition function that selects the point with the
    highest probability of yielding a value lower than the current best f_best.
    
    For a Gaussian posterior N(mu, sigma^2), the PI is:
    PI = P(f < f_best) = Phi((f_best - mu) / sigma)
    where z = (f_best - mu) / sigma.
    
    This approach is distinct from EI as it does not consider the magnitude of
    improvement, only the likelihood of improvement. It can be more robust to
    scale variations and less prone to over-exploiting high-magnitude but
    low-probability regions.
    """
    # Ensure sigma is positive and handle near-zero values
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / safe_sigma
    
    # Calculate PI
    # norm.cdf is Phi
    pi = norm.cdf(z)
    
    return pi
# EVOLVE-BLOCK-END
