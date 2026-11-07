"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Draws a single sample from the posterior Gaussian distribution
    and scores candidates by the negative of this sample. This naturally
    balances exploration and exploitation by probabilistically selecting
    regions with high potential for the global minimum.
    
    Parameters:
    mu (np.array): Posterior mean at candidate points, shape (N,)
    sigma (np.array): Posterior standard deviation at candidate points, shape (N,)
    f_best (float): Current best observed value
    
    Returns:
    np.array: Scores for each candidate, shape (N,). Higher is better.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: simple UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: draw a sample from the posterior
    # y_sample = mu + sigma * randn
    randn = np.random.randn(n)
    y_sample = mu + sigma * randn
    
    # Score is negative of the sample (lower y_sample is better, so higher score is better)
    score = -y_sample
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
