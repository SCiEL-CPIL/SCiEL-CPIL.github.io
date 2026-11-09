"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid acquisition: blends Expected Improvement with UCB exploration,
    normalized by cost to prefer cheaper evaluations.
    
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
    
    # UCB component: f_best - mu + beta * sigma
    # This rewards high uncertainty (exploration) even when mean is worse than f_best
    beta = 1.5
    ucb = (f_best - mu) + beta * sigma
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Blend EI and UCB: use a weighted combination
    # EI is more conservative (only positive when improvement expected),
    # UCB is more aggressive (always positive for high sigma)
    # Weight: 0.5 EI + 0.5 UCB gives balanced exploration/exploitation
    blended = 0.5 * ei + 0.5 * ucb
    blended = np.where(np.isfinite(blended), blended, 0.0)
    
    # Cost-aware scaling: divide by cost to prefer cheaper evaluations
    scores = blended / cost
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores (shouldn't be needed but safe)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
