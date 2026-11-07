import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Powell 4D BO.
    Uses Upper Confidence Bound (UCB) divided by Cost.
    """
    elapsed = _get_elapsed()
    if elapsed > 280:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-6)
        # Fallback UCB
        beta = 2.0
        score = (-mu + beta * sigma) / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Cost-aware UCB
    # For minimization, we want to minimize mu - beta * sigma.
    # To maximize the acquisition score: -mu + beta * sigma.
    beta = 2.0
    cost_safe = np.maximum(cost, 1e-6)
    
    score = (-mu + beta * sigma) / cost_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
