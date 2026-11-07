"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Thompson Sampling acquisition: sample from posterior, return negative sample
# This provides stochastic exploration that can escape local optima better than EI
# for multi-modal functions like Ackley.

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Posterior mean at candidate points (in objective units)
    sigma : array-like, shape (n,)
        Posterior standard deviation at candidate points (in objective units)
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : array, shape (n,)
        Higher scores mean more likely to be selected next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    
    # Draw a single standard normal sample for each candidate
    # theta = mu + sigma * Z, where Z ~ N(0, 1)
    # We want to minimize, so we return -theta (higher = better)
    rng = np.random.default_rng()
    z = rng.standard_normal(mu.shape)
    theta = mu + sigma * z
    scores = -theta
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
