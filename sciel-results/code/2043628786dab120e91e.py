"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 280s safety margin (limit is 300s)


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and UCB with adaptive exploration.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: prefer low cost and low mu
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Theoretical minimum for Styblinski-Tang 2D
    f_min_theoretical = -78.3323
    
    # Heuristic start value for progress calculation
    f_start = 20.0
    
    # Calculate progress in [0, 1]
    # 0.0 = no progress (f_best >= f_start)
    # 1.0 = full progress (f_best <= f_min_theoretical)
    denom = f_start - f_min_theoretical
    progress = np.clip((f_start - f_best) / denom, 0.0, 1.0)
    
    # Adaptive beta for UCB:
    # Starts at 2.5 (high exploration) and decreases to 0.8 (low exploration)
    beta_min = 0.8
    beta_max = 2.5
    beta = beta_min + (beta_max - beta_min) * (1.0 - progress)
    
    # Compute Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI can be negative if mu > f_best; clip to 0 for stability
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    score_ei = ei / cost
    
    # UCB for minimization:
    # Lower confidence bound: mu - beta * sigma
    # We want to select the point with the lowest lower bound.
    # To fit "higher score = better", we negate: score = (beta * sigma - mu) / cost
    score_ucb = (beta * sigma - mu) / cost
    
    # Combine EI and UCB with adaptive weights
    # Early on: more weight on UCB (exploration)
    # Later: more weight on EI (exploitation)
    w_ucb = 0.3 + 0.4 * (1.0 - progress)  # ranges from 0.3 to 0.7
    w_ei = 1.0 - w_ucb
    w_ei = np.maximum(w_ei, 0.1)  # keep at least 10% EI
    
    # Normalize weights
    w_sum = w_ei + w_ucb
    w_ei_norm = w_ei / w_sum
    w_ucb_norm = w_ucb / w_sum
    
    score = w_ei_norm * score_ei + w_ucb_norm * score_ucb
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
