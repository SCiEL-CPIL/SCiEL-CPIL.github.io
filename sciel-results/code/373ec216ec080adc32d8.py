"""Rewritten program with Cost-Aware UCB acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_TIME_LIMIT = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are too close to the limit, return a safe fallback
    # (e.g., uniform scores or zero, forcing the evaluator to pick arbitrarily or the first one)
    # Since higher is better, returning zeros is safe.
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros(len(mu))

    # Cost-Aware UCB for minimization
    # We want to select x that minimizes f(x).
    # UCB value for minimization: mu - beta * sigma
    # We want to MAXIMIZE the returned score.
    # Score = (beta * sigma - mu) / cost
    
    # Tuning parameter for exploration
    beta = 2.5
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate scores
    scores = (beta * sigma - mu) / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
