# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using UCB.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points
    f_best : float
        Best observed value so far (minimum)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better)
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 300s limit
    
    # Handle edge cases: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # UCB for minimization:
    # We want to find points with low expected value (low mu)
    # and high uncertainty (high sigma).
    # Score = -mu + beta * sigma
    # Higher score means better candidate.
    
    # Choose beta. 
    # A common choice is beta = 1.0 to 3.0.
    # Let's use 2.0 as a balanced exploration/exploitation tradeoff.
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time before returning
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
