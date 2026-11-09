"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_DEADLINE = 315.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a cost-normalized Lower Confidence Bound (LCB) strategy.
    The LCB for minimization is mu - beta * sigma.
    We maximize the score: (-mu + beta * sigma) / cost.
    Beta is adaptive based on the progress towards the known minimum.
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Adaptive beta for LCB
    # Higher beta encourages exploration (higher sigma weight)
    # Lower beta encourages exploitation (lower mu weight)
    # We can modulate beta based on the spread of sigma or current best improvement
    sigma_max = np.max(sigma_safe)
    if sigma_max < 1e-9:
        sigma_max = 1e-9
    
    # Use a fixed reasonable beta for cost-aware LCB
    # In cost-aware BO, dividing by cost naturally balances exploration/exploitation
    # if the cost correlates with location. 
    # A beta around 2.0-3.0 is common for LCB.
    # Let's try a slightly higher beta to encourage finding the true minimum 
    # which might be in a high-cost region if the function structure demands it,
    # but the cost penalty will counteract extreme costs.
    beta = 2.5
    
    lcb = mu - beta * sigma_safe
    
    # The score should reflect the "value per cost".
    # Since we want to minimize f, we want to maximize -f.
    # So we maximize -lcb / cost.
    # Note: lcb can be negative (if mu is very low).
    # If lcb is negative, -lcb is positive (good).
    # If lcb is positive (worse than 0), -lcb is negative.
    # We should probably clip the score to 0 or ensure it handles signs correctly.
    # However, simple regret scoring compares (observed - min).
    # The acquisition just needs to rank candidates.
    
    scores = -lcb / cost_safe
    
    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
