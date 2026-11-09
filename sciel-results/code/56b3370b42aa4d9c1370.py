"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Expected Improvement (EI) with a small exploration bonus based on uncertainty.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) acquisition function
    # We are minimizing, so improvement is (f_best - mu)
    z = (f_best - mu) / sigma
    # Use scipy.stats.norm for cdf and pdf
    from scipy.stats import norm
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration term to encourage exploring high-uncertainty regions
    # This helps escape local minima in the Goldstein-Price function
    # The exploration term is proportional to sigma, weighted by a small factor
    # We use a fixed exploration coefficient
    beta = 0.5
    score = ei + beta * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
