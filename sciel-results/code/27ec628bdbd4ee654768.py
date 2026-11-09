"""Acquisition function for Bayesian optimization of weierstrass 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using Upper Confidence Bound (UCB).
    
    Args:
        mu: array of shape (2000,) with posterior means.
        sigma: array of shape (2000,) with posterior standard deviations.
        f_best: scalar, the lowest observed value so far.
        
    Returns:
        array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # UCB for minimization: maximize (beta * sigma - mu)
    # beta controls the exploration/exploitation trade-off
    # A higher beta encourages more exploration
    # Previous attempt with beta=2.0 scored 0.79.
    # EI scored 1.78.
    # Let's try beta=1.0 which is a standard balanced choice.
    beta = 1.0
    
    scores = beta * sigma - mu
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
