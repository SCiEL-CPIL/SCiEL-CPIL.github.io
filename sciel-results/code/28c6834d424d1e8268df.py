"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the time limit, return a simple exploitation score
    if time.time() > _DEADLINE:
        # Simple exploitation: prefer points with lowest mu
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Cost-aware POI with dynamic improvement threshold
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Estimate remaining budget: total budget is 30, cost is the cumulative cost
    # We need to estimate how many evaluations remain
    # Average cost per eval is roughly 1.5 (mean of 1 + mean(x) where x in [0,1])
    # So remaining evals ~ (30 - cost) / 1.5
    remaining_budget = np.maximum(30.0 - cost, 0.0)
    # Estimate remaining evaluations
    avg_cost = 1.5
    n_remaining = np.maximum(remaining_budget / avg_cost, 1.0)
    
    # Dynamic gamma: as budget depletes, demand larger improvements
    # gamma scales the threshold based on remaining budget
    # When budget is full, gamma is small (more exploration)
    # When budget is low, gamma is larger (more exploitation)
    gamma = 0.1 + 0.5 * (1.0 - remaining_budget / 30.0)
    
    # Dynamic threshold: target improvement scales with remaining budget
    # As budget depletes, we demand larger relative improvements
    threshold = gamma * (30.0 - cost) / np.maximum(n_remaining, 1.0)
    
    # POI with dynamic threshold: P(f < f_best - threshold)
    # = P(Z < (f_best - threshold - mu) / sigma)
    z = (f_best - threshold - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Ensure POI is in [0, 1]
    poi = np.clip(poi, 0.0, 1.0)
    
    # Divide by cost to account for evaluation cost
    cost_safe = np.maximum(cost, 1e-8)
    scores = poi / cost_safe
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
