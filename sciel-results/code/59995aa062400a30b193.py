import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 290:
        z = (f_best - mu) / (sigma + 1e-9)
        EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        EI = np.maximum(EI, 0)
        return EI / (cost + 1e-9)

    # Compute Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    EI = np.maximum(EI, 0.0)
    
    # Compute cost savings relative to max cost
    max_cost = np.max(cost)
    cost_savings = max_cost - cost
    
    # Pareto frontier filter:
    # A candidate is Pareto-optimal if no other candidate has both
    # higher EI and lower cost (or equal in one and better in the other)
    # We'll compute this efficiently by sorting by cost and keeping
    # candidates with strictly increasing EI
    
    # Sort by cost ascending
    sorted_idx = np.argsort(cost)
    sorted_cost = cost[sorted_idx]
    sorted_EI = EI[sorted_idx]
    
    # Keep only Pareto-optimal: EI must be strictly greater than all previous
    # (since cost is already sorted ascending)
    pareto_mask = np.zeros(len(sorted_EI), dtype=bool)
    max_EI_so_far = -np.inf
    for i in range(len(sorted_EI)):
        if sorted_EI[i] > max_EI_so_far:
            pareto_mask[i] = True
            max_EI_so_far = sorted_EI[i]
    
    # Map back to original indices
    pareto_original = np.zeros(len(EI), dtype=bool)
    pareto_original[sorted_idx[pareto_mask]] = True
    
    # For non-Pareto candidates, assign a low score
    # For Pareto candidates, compute hypervolume-like contribution
    # Hypervolume contribution: EI * cost_savings (multiplicative)
    # This rewards candidates that offer both high improvement and low cost
    
    scores = np.zeros_like(EI)
    scores[pareto_original] = EI[pareto_original] * cost_savings[pareto_original]
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
