import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float, lowest observed value
    
    Returns:
    score: array of shape (2000,)
    """
    # Ensure sigma is strictly positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # UCB for minimization:
    # We want to maximize the score.
    # A low predicted value (mu) is good. So we use -mu.
    # High uncertainty (sigma) is good for exploration. So we add kappa * sigma.
    kappa = 1.5
    score = -mu + kappa * sigma
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline Guard
    elapsed = time.time() - _start_time
    if elapsed > 280: # Safety margin before 300s limit
        # If we are running out of time, we still return the computed scores.
        # There is no "best result found" to return differently, just the scores.
        pass
        
    return score
