"""Cost-aware acquisition for Bayesian optimization on 4 test functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    - mu: (2000,) array of GP means
    - sigma: (2000,) array of GP standard deviations
    - f_best: scalar, best observed value
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - (2000,) array of scores (higher = more desirable)
    """
    # Check deadline - if we're running out of time, return a safe fallback
    if _time_remaining() < 5.0:
        # Fallback: pure uncertainty-based to be fast
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return sigma_safe / cost_safe
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # UCB-style exploration term
    # Use a beta that balances exploitation and exploration
    # As we get closer to the minimum, we want more exploitation
    # We can estimate progress by how close f_best is to 0
    # But we don't know the minimum exactly, so use a heuristic
    
    # Cost-aware scoring: use quadratic cost penalty to strongly favor low-cost evaluations
    # This allows more evaluations within the fixed budget of 30
    cost_penalty = 1.0 / (cost ** 2)
    
    # Primary score: EI with quadratic cost penalty
    score_ei = ei * cost_penalty
    
    # Exploration score: uncertainty with quadratic cost penalty
    score_explore = sigma * cost_penalty
    
    # Combine: primarily EI, with some exploration
    # The balance can be adaptive based on how much improvement we've seen
    # For now, use a fixed weight that favors exploitation slightly
    w_explore = 0.15
    score = (1.0 - w_explore) * score_ei + w_explore * score_explore
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
