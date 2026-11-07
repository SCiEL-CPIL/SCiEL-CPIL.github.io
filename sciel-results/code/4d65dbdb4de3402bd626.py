"""Rewritten program with cost-aware UCB acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    For minimization, we score points based on the potential to be lower than f_best,
    adjusted for uncertainty (sigma) and divided by cost.
    Score = (f_best - mu + beta * sigma) / cost
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value
    cost : ndarray, shape (2000,)
        Cost of evaluation
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores
    """
    start_time = time.time()
    deadline = start_time + 290
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # UCB for minimization:
    # We want to maximize (f_best - mu + beta * sigma) / cost
    # beta controls exploration vs exploitation.
    # Knowledge suggests beta=2.
    beta = 2.0
    
    numerator = (f_best - mu) + beta * sigma
    scores = numerator / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    if time.time() > deadline:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
