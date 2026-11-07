"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for deadline tracking and call counting
_start_time = None
_deadline = None
_call_count = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) with
    adaptive cost-awareness.
    
    Structural change from PI: UCB explicitly balances exploration (via
    uncertainty) and exploitation (via mean prediction), which is more
    aligned with minimizing simple regret than pure probability of improvement.
    
    The cost-awareness is implemented via an adaptive exponent on cost:
    - Early in the run (plentiful budget): alpha = 1.0 (standard cost-awareness)
    - Late in the run (tight budget): alpha increases to favor cheap evaluations
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: scalar - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time, _deadline, _call_count
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin before 330s limit
    
    # Increment call count
    _call_count += 1
    
    # Deadline guard: return simple fallback if time is running out
    if time.time() > _deadline:
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        cost_safe = np.maximum(cost, eps)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        scores = pi / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute Z-scores
    z = (f_best - mu) / sigma_safe
    
    # Upper Confidence Bound: UCB = mu + kappa * sigma
    # Since we're minimizing, we want to find low values, so we use:
    # UCB_min = mu - kappa * sigma (lower bound with uncertainty)
    # But for acquisition, higher score = better, so:
    # We want to maximize the probability of finding a low value.
    # Standard UCB for minimization: score = -mu + kappa * sigma
    # Or equivalently, we can use: score = (f_best - mu) + kappa * sigma
    # which rewards points with low predicted values and high uncertainty.
    
    # Adaptive kappa: start with moderate exploration, increase as budget depletes
    # to be more aggressive in finding cheap improvements
    
    # Estimate remaining budget
    # Initial 5 points: ~7.5 cost (uniform in [0,1]^6, mean cost = 1 + 0.5 = 1.5 each)
    # Each subsequent point: ~1.5 cost (average)
    initial_cost_est = 7.5
    avg_cost_per_step = 1.5
    total_cost_est = initial_cost_est + _call_count * avg_cost_per_step
    remaining_budget = max(0.0, 30.0 - total_cost_est)
    
    # Adaptive exploration parameter kappa
    # When budget is plentiful, moderate exploration (kappa ~ 1.0)
    # When budget is tight, more aggressive exploration of cheap points
    if remaining_budget > 10.0:
        kappa = 1.0
    elif remaining_budget < 5.0:
        kappa = 2.0
    else:
        kappa = 1.0 + (10.0 - remaining_budget) / 5.0
    
    # UCB score for minimization:
    # We want to find points likely to be below f_best
    # score = (f_best - mu) + kappa * sigma
    # This is high when mu is low (exploit) or sigma is high (explore)
    ucb_score = (f_best - mu) + kappa * sigma
    
    # Cost-awareness: normalize by cost with adaptive exponent
    # Early: alpha = 1.0, Late: alpha increases to favor cheap evaluations
    if remaining_budget > 10.0:
        alpha = 1.0
    elif remaining_budget < 5.0:
        alpha = 1.5
    else:
        alpha = 1.0 + 0.5 * (10.0 - remaining_budget) / 5.0
    
    # Final score: UCB / (cost ** alpha)
    # Clip UCB to non-negative to avoid negative scores for very expensive points
    ucb_score = np.maximum(ucb_score, 0.0)
    scores = ucb_score / (cost_safe ** alpha)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
