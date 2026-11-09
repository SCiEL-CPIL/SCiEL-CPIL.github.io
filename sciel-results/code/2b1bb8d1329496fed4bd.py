"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using multiplicative hypervolume scoring:
    EI * cost_savings, where cost_savings = (max_cost - cost) / max_cost.
    This rewards both improvement potential and cost efficiency multiplicatively,
    which can escape local minima more effectively than division.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the 330s limit, return a simple safe result
    if time.time() - _start_time > 310:
        z = (f_best - mu) / (sigma + 1e-9)
        z = np.clip(z, -10, 10)
        EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        EI = np.maximum(EI, 0.0)
        scores = EI / (cost + 1e-9)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Cost savings: fraction of maximum possible cost saved
    # Cost ranges from 1 to 2 (since x scaled to [0,1], mean is in [0,1], so cost in [1,2])
    # cost_savings = (2 - cost) / 1, clamped to [0, 1]
    max_cost = 2.0
    min_cost = 1.0
    cost_savings = (max_cost - cost) / (max_cost - min_cost)
    cost_savings = np.clip(cost_savings, 0.0, 1.0)
    cost_savings = np.where(np.isfinite(cost_savings), cost_savings, 0.5)
    
    # Multiplicative hypervolume scoring: EI * cost_savings
    # This creates a different trade-off surface than division
    scores = ei * cost_savings
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
