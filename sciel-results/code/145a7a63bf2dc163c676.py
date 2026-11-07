"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 330s wall-clock.
# Track elapsed time and return early if approaching the limit.
_start_time = [time.time()]
_DEADLINE = 325.0  # 5s safety margin

def _elapsed():
    return time.time() - _start_time[0]

def _reset_deadline():
    _start_time[0] = time.time()

def _maybe_reset_deadline():
    # If elapsed time is very small (first call or just reset), keep it
    if _elapsed() < 0.005:
        _start_time[0] = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized UCB for minimization with high-gamma cost penalty.
    
    score = (mu + kappa * sigma) / cost^gamma
    
    For minimization, we want to find points with low predicted value (mu)
    and high uncertainty (sigma). The cost^gamma term heavily penalizes
    expensive evaluations, directing search to cheap, unexplored regions.
    
    Higher score = more desirable to evaluate next.
    """
    # Deadline check: if we're running out of time, return a safe fallback
    if _elapsed() > _DEADLINE:
        # Return scores that favor the cheapest, most uncertain points
        # as a safe fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-6)
        fallback = (mu + 2.0 * sigma_safe) / (cost_safe ** 3.0)
        fallback = np.nan_to_num(fallback, nan=0.0, posinf=1e6, neginf=-1e6)
        return fallback

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=0.0)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Kappa controls exploration-exploitation balance
    # Higher kappa = more exploration
    kappa = 2.0
    
    # Gamma controls how aggressively we penalize expensive points
    # Higher gamma = stronger preference for cheap regions
    gamma = 3.0
    
    # UCB value for minimization: we want low mu + high sigma
    # The quantity (mu + kappa * sigma) is high where we expect
    # either good predictions or high uncertainty
    ucb_values = mu + kappa * sigma
    
    # Cost-normalized score: divide by cost^gamma to heavily penalize
    # expensive evaluations. This directs search to cheap, unexplored regions.
    scores = ucb_values / (cost ** gamma)
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
