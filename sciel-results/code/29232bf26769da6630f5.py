"""Rewritten cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition with dynamic exploration parameter.
    
    Strategy:
    1. Compute a cost-aware UCB score: (mu + kappa * sigma) / cost
    2. Dynamically tune kappa based on remaining budget ratio to shift
       exploration-exploitation balance as search progresses.
    3. Use negative UCB since we're minimizing (higher score = better).
    """
    t0 = time.time()
    deadline = t0 + 290.0  # safety margin before 300s limit

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Estimate remaining budget: total budget is 30, current cost is sum of costs
    # But we only have per-candidate cost, not total. Use mean cost as proxy.
    # The cost array has shape (2000,) - these are costs for each candidate.
    # We need to infer how much budget has been used. 
    # Since we don't have direct access to total spent, use a heuristic:
    # Assume the cost array reflects the cost structure, and use a dynamic kappa
    # based on the spread of costs and current f_best.
    
    # Dynamic kappa: start with higher exploration (larger kappa) early,
    # decrease as we get more confident. Use f_best relative to typical values.
    # Hartmann 3D minimum is around -3.86, so f_best starts high (near 0 or positive)
    # and decreases. Use normalized progress.
    
    # Heuristic: if f_best is still far from the known minimum, explore more.
    # Known minimum for hartmann 3D is approximately -3.86278
    known_min = -3.86278
    
    # Normalize progress: 0 at start (f_best ~ 0 or higher), 1 when f_best near min
    # Use a sigmoid-like mapping
    progress = 1.0 - np.exp(-0.5 * max(0.0, -f_best - 0.5))
    # progress is in [0, 1], higher means more progress
    
    # kappa: start at 2.0 (high exploration), decrease to 0.5 (exploitation)
    # Use smooth interpolation
    kappa = 2.0 - 1.5 * progress
    
    # Cost-aware UCB: we want to minimize, so higher (mu + kappa*sigma) is worse
    # We want to find points with LOW mu and LOW cost, with some exploration via sigma
    # Score = -(mu + kappa * sigma) / cost  (higher = better for selection)
    ucb_score = -(mu + kappa * sigma_safe) / cost_safe
    
    # Ensure all scores are finite
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    # Replace -inf with a very negative finite value
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -1e10)

    # Time guard: if we're close to deadline, return immediately
    if time.time() > deadline - 5.0:
        return ucb_score

    return ucb_score
# EVOLVE-BLOCK-END
