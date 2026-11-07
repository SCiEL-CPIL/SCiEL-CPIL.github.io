"""Cost-aware LCB acquisition for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound acquisition function for Bayesian optimization.
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
    
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    N = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Dynamic exploration coefficient beta:
    # Start with moderate exploration, scale with uncertainty relative to f_best
    # Higher uncertainty or larger gap from f_best -> more exploration
    sigma_mean = np.mean(sigma_safe)
    gap = max(f_best, 1e-6)  # Avoid division by zero if f_best is near 0
    # Beta scales with normalized uncertainty and decreases as we get closer to optimum
    beta = 2.0 + 1.5 * (sigma_mean / (gap + 1e-6))
    
    # LCB: we want to minimize, so lower is better
    # LCB = mu - beta * sigma (this is what we want to minimize)
    # But for acquisition (higher = better), we use: -LCB = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost to prioritize cheap evaluations
    cost_safe = np.maximum(cost, 1e-10)
    scores = lcb_score / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline - if so, return current result
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
