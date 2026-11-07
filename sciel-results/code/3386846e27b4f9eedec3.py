"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _check_deadline():
    return time.time() > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Computes the acquisition score for each candidate.
    Uses cost-normalized Lower Confidence Bound (LCB).
    """
    if _check_deadline():
        return np.zeros_like(mu)
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Dynamic beta scaling: encourages exploration early, exploitation later
    # As f_best decreases (improves), we can be more conservative
    # Use a simple heuristic: beta scales with the magnitude of uncertainty relative to f_best
    eps = 1e-9
    
    # Compute the LCB: f_best - mu + beta * sigma
    # We want to maximize (f_best - mu + beta * sigma) / cost
    # which is equivalent to minimizing mu - beta*sigma + cost adjustment
    
    # Dynamic beta: start high for exploration, decrease as we find better solutions
    # Use a simple scaling based on the spread of sigma
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Beta should be large when uncertainty is high relative to the current best
    # A common choice is beta = log(n * pi^2 / 6) but we don't have n directly
    # Let's use a fixed but reasonable beta that balances exploration/exploitation
    # For Rastrigin, we need good exploration
    beta = 2.0
    
    # LCB score: (f_best - mu + beta * sigma) / cost
    # This is the expected improvement in the LCB sense, normalized by cost
    lcb_values = f_best - mu + beta * sigma
    
    # We want to maximize lcb_values / cost
    # But we need to ensure we only consider points that have a chance of improvement
    # lcb_values > 0 means the lower bound is better than f_best
    
    scores = lcb_values / (cost + eps)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
