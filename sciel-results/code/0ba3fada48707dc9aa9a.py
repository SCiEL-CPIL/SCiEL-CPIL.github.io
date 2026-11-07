import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates. Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Fallback to pure exploitation to save time
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e6)
        return scores

    # UCB parameter
    # Higher beta encourages exploration
    beta = 4.0
    
    # Score = beta * sigma - mu
    # Maximizing this is equivalent to minimizing (mu - beta * sigma)
    scores = beta * sigma - mu
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
