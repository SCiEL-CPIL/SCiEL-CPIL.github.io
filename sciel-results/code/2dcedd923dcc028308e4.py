import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return exploit-heavy fallback
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, 1e6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive LCB: balance exploitation (low mu) and exploration (high sigma)
    # beta controls exploration-exploitation tradeoff, decreasing over time
    if _call_count <= 5:
        beta = 0.25
    elif _call_count <= 10:
        beta = 0.15
    elif _call_count <= 20:
        beta = 0.08
    elif _call_count <= 30:
        beta = 0.04
    else:
        beta = 0.02
    
    # LCB for minimization: prefer low mean + uncertainty bonus
    lcb = mu - beta * sigma
    
    # Cost-aware LCB: normalize by cost to prefer cheaper evaluations
    # Use a power less than 1 to soften cost penalty (encourages some expensive high-value points)
    cost_aware_lcb = lcb / (cost_safe ** 0.7)
    
    # Add small exploration bonus for very uncertain regions (early on)
    if _call_count <= 15:
        uncertainty_bonus = 0.01 * sigma / cost_safe
    else:
        uncertainty_bonus = 0.0
    
    scores = cost_aware_lcb + uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
