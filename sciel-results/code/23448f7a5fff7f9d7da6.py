import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: numpy array of shape (2000,), mean of the GP posterior.
        sigma: numpy array of shape (2000,), standard deviation of the GP posterior.
        f_best: float, the best (lowest) observed function value so far.
        
    Returns:
        numpy array of shape (2000,), scores where higher is better.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Exploit the best mean (greedy)
        # Higher score means we want to pick it.
        # Since we want to minimize, lower mu is better.
        # So score = -mu
        return -mu.copy()

    # UCB for minimization:
    # We want to select points that have a low predicted mean AND high uncertainty.
    # Score = -mu + beta * sigma
    # If mu is low (good), -mu is high (good score).
    # If sigma is high (uncertain), the score increases (exploration).
    
    # Determine beta.
    # A common choice is beta = 2 or 3 for short horizons.
    # We can also make it slightly adaptive based on f_best, but for simplicity and robustness,
    # a fixed value is often stable in 1D.
    # Let's use beta = 2.5.
    beta = 2.5
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate UCB scores
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
