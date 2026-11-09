"""Cost-aware acquisition: adaptive cost-weighted EI/UCB blend with model-based cost penalty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s wall-clock limit per evaluation; use safety margin

def _elapsed_ok():
    """Return True if we have at least ~5s margin before the hard limit."""
    return (time.time() < _DEADLINE - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the limit, return a simple score quickly
    if not _elapsed_ok():
        sigma_safe = np.maximum(sigma, 1e-6)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Robust sigma: ensure it's positive and not too small
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    # Handle cases where z is very large (EI ~ f_best - mu) or very negative (EI ~ 0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Make EI non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    ei_cost = ei / cost_safe
    
    # UCB: -mu + kappa * sigma (negative because we minimize)
    ucb = -mu + 2.0 * sigma_safe
    
    # Normalize both to [0, 1]
    ei_c_min = np.min(ei_cost)
    ei_c_max = np.max(ei_cost)
    ei_c_range = ei_c_max - ei_c_min
    if ei_c_range > 1e-12:
        ei_c_norm = (ei_cost - ei_c_min) / ei_c_range
    else:
        ei_c_norm = np.ones_like(ei_cost)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Adaptive weighting: learn the cost penalty from observed cost distribution
    # Higher cost candidates should be penalized more strongly
    cost_mean = np.mean(cost_safe)
    cost_std = np.std(cost_safe)
    
    # If cost varies significantly, increase the weight on cost-aware EI
    if cost_std > 0.01:
        # Cost varies a lot: favor cost-aware EI more
        alpha = 0.75
    else:
        # Cost is relatively uniform: balance exploration and exploitation
        alpha = 0.6
    
    # Weighted combination: adaptive alpha for cost-aware EI, (1-alpha) for UCB
    score = alpha * ei_c_norm + (1 - alpha) * ucb_norm
    
    # Additional cost penalty: multiply by a cost-aware factor that downweights expensive candidates
    # This creates a smooth penalty that is stronger for higher-cost candidates
    cost_penalty = 1.0 / (1.0 + 0.5 * (cost_safe - cost_mean) / (cost_std + 1e-9))
    score = score * cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
