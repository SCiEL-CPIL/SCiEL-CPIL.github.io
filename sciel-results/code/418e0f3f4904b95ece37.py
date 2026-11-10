"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Optimistic UCB per unit cost.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard: if we're running low on time, return a safe fallback
    if time.time() - _start_time > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-12)
        cost_safe = np.maximum(cost, 1e-12)
        # Optimistic bound: f_best - mu + kappa * sigma
        kappa = 1.0
        score = (f_best - mu + kappa * sigma_safe) / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Numerical stability
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Optimistic UCB for minimization:
    # We want to maximize the potential improvement.
    # A point is promising if its mean is low (f_best - mu is large)
    # or its uncertainty is high (sigma is large).
    # Score = (f_best - mu) + kappa * sigma
    kappa = 1.0
    score = (f_best - mu + kappa * sigma_safe) / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
