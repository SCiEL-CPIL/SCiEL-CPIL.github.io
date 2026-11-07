"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Draw a single sample from the posterior at each candidate point:
        f_s = mu + sigma * xi,  xi ~ N(0, 1)
    
    Return -f_s as the score (higher is better for selection).
    This stochastic approach naturally balances exploration and exploitation.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, np.nanmin(mu[np.isfinite(mu)]) if np.any(np.isfinite(mu)) else 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    
    # Draw a single standard normal sample for all 2000 candidates
    xi = np.random.standard_normal(mu.shape)
    
    # Posterior sample
    f_s = mu + sigma * xi
    
    # Score: higher is better, so we negate the sampled objective value
    score = -f_s
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
