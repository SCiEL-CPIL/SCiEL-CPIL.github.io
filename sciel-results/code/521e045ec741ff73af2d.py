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
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    cost_aware_ei = ei / cost_safe
    
    # Adaptive exploration bonus based on call count
    # Tuned to balance early exploration for multi-modal Shekel 4D and late exploitation
    if _call_count <= 5:
        exploration_factor = 0.12
    elif _call_count <= 10:
        exploration_factor = 0.08
    elif _call_count <= 20:
        exploration_factor = 0.05
    else:
        exploration_factor = 0.02
    
    # Uncertainty bonus scaled by cost
    exploration_bonus = sigma / cost_safe * exploration_factor
    
    # Combine EI and exploration
    scores = cost_aware_ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
