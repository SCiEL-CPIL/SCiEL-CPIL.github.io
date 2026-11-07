"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition with adaptive exploration.
    
    Uses a Lower Confidence Bound (optimistic prediction) scaled by an
    adaptive beta that depends on both the remaining budget (via cost)
    and the current best value. The cost is combined convexly to avoid
    over-penalizing high-uncertainty regions early.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-6)
    
    # Adaptive beta: increases exploration when f_best is far from 0 (the known minimum)
    # and when cost is low (early in the budget). 
    # f_best is the lowest observed value; for three hump camel, min is ~0.
    # Use a scale relative to the initial range of the function.
    # Three hump camel on [-5,5]^2 has values roughly in [-2.9, 2.9]
    # f_best starts around 1-2 and decreases toward 0.
    
    # Normalized progress: how close are we to the minimum?
    # Assume initial f_best is around 1.5, final target is 0.
    # progress = 1 - f_best / initial_estimate
    initial_estimate = 2.0  # typical initial f_best
    progress = np.clip(1.0 - f_best / initial_estimate, 0.0, 1.0)
    
    # Beta decreases as we make progress (exploit more late)
    # Beta also depends on cost: low cost = early = explore more
    mean_cost = np.mean(cost)
    # Normalize cost: cost ranges from 1 to 2 (since x in [0,1] scaled, cost = 1 + mean(x))
    # Early: cost ~ 1, Late: cost ~ 1.5
    cost_factor = 1.0 + 2.0 * (1.0 - (mean_cost - 1.0))  # higher when cost is low (early)
    
    # Base beta
    beta = 2.0 * cost_factor * (1.0 - 0.5 * progress)
    # Ensure beta is reasonable
    beta = np.clip(beta, 0.5, 5.0)
    
    # LCB: mu - beta * sigma (optimistic prediction, lower is better)
    lcb = mu - beta * sigma
    
    # Cost-aware combination: we want to minimize lcb but also prefer low cost.
    # Use a convex combination that balances improvement and cost.
    # score = -lcb / cost_effective
    # But to avoid over-penalizing early, use a softer cost penalty.
    # cost_effective = cost^gamma, where gamma < 1 reduces the penalty impact
    gamma = 0.7  # sublinear cost penalty
    
    cost_eff = cost ** gamma
    
    # Higher score = better. We want low lcb (good) and low cost (good).
    # score = -lcb / cost_eff
    # But lcb can be negative, so -lcb is positive when lcb is negative (good).
    # To ensure finite and positive scores, add a small offset.
    
    # Alternative: use a weighted sum in the transformed space
    # Normalize lcb and cost to [0, 1] range for this batch
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min + 1e-9
    lcb_norm = (lcb - lcb_min) / lcb_range  # 0 = best, 1 = worst
    
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Weighted sum: prefer low lcb_norm and low cost_norm
    # Weight shifts from exploration (lcb) to exploitation (cost) over time
    # Early: weight more on lcb (explore), Late: weight more on cost (exploit cheap good points)
    w_lcb = 0.7 + 0.3 * progress  # 0.7 early, 1.0 late
    w_cost = 1.0 - w_lcb + 0.3  # keep some cost weight always
    
    # Normalize weights
    w_sum = w_lcb + w_cost
    w_lcb /= w_sum
    w_cost /= w_sum
    
    # Score: higher is better
    # Low lcb_norm is good, low cost_norm is good
    score = w_lcb * (1.0 - lcb_norm) + w_cost * (1.0 - cost_norm)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
