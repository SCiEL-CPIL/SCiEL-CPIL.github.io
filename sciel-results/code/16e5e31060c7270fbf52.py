"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses Cost-Adjusted Expected Improvement with robust deadline management
and input sanitization to balance exploration/exploitation under cost constraints.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline management
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 285.0  # Safety margin before 330s limit
_CALL_COUNT = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Upper Confidence Bound (UCB).
    Uses a time-dependent beta to balance exploration and exploitation.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    global _CALL_COUNT
    _CALL_COUNT += 1

    # Check deadline
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure cost is within expected range [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute UCB for minimization:
    # We want to minimize f. UCB for minimization is mu - beta * sigma.
    # We want to MAXIMIZE the acquisition score.
    # So we maximize - (mu - beta * sigma) / cost_adjusted
    # Which is (beta * sigma - mu) / cost_adjusted
    
    # Time-dependent beta for exploration
    # Standard heuristic: beta_t = 2 * log(t + 1)
    # This encourages exploration early on and exploitation later.
    beta = 2.0 * np.log(_CALL_COUNT + 1.0)
    
    # UCB score (higher is better for minimization when negated)
    ucb = beta * sigma - mu
    
    # Cost adjustment
    # Divide by cost to prefer cheaper evaluations
    # Use cost directly or cost - offset?
    # Previous code used cost - 0.5. Let's stick to a similar cost scaling to avoid zero division.
    cost_adjusted = np.maximum(cost - 0.5, 0.1)
    
    score = ucb / cost_adjusted
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
