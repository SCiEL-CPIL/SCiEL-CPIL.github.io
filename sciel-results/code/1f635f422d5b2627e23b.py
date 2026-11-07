"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We use a 325s deadline to be safe.
_start_time = time.time()
_DEADLINE = 325.0


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Computes the acquisition score for each candidate.
    Uses a hybrid approach:
    1. If deadline is near, return a simple deterministic score.
    2. Otherwise, use a cost-aware Lower Confidence Bound (LCB) with
       dynamic beta scaling to balance exploration and exploitation.
    3. Combine with a Probability of Improvement (PoI) component
       to ensure exploitation of promising regions.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Deadline guard: if we're running out of time, return a simple score
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return (f_best - mu) / cost_safe
    
    # Safe versions to prevent division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate z-score for PoI
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Cost-aware score: PoI / adjusted_cost
    # Compress cost range to reduce penalty for high-cost points
    # Cost is in [1, 2]. Adjusted is in [1, 1.5]
    adjusted_cost = 0.5 * cost_safe + 0.5
    poi_score = poi / (adjusted_cost + eps)
    
    # LCB component: (f_best - mu + beta * sigma) / cost
    # Dynamic beta: scale by uncertainty to encourage exploration
    # Use mean sigma as a reference for scaling
    mean_sigma = np.mean(sigma_safe)
    if mean_sigma < eps:
        mean_sigma = eps
    
    # Beta that balances exploration and exploitation
    # Higher beta encourages more exploration
    beta = 1.5
    
    # LCB score
    lcb_score = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Combine PoI and LCB scores
    # Weight PoI more heavily for exploitation, LCB for exploration
    weight_poi = 0.6
    weight_lcb = 0.4
    
    scores = weight_poi * poi_score + weight_lcb * lcb_score
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
