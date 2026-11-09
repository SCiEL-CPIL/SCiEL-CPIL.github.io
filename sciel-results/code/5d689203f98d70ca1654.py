"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using probability-of-improvement (PI) with
    soft cost normalization based on the empirical cost distribution.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma has a small floor to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    # z = (f_best - mu) / sigma
    # PI = Phi(z)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    pi = norm.cdf(z)
    
    # Clip to [0, 1]
    pi = np.clip(pi, 0.0, 1.0)
    
    # Soft cost normalization: instead of linear penalty, use a softmax-like
    # weighting that down-weights high-cost points based on the cost distribution.
    # This adapts to the actual cost landscape rather than assuming a fixed penalty.
    
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute a soft cost weight: exp(-beta * (cost - cost_min) / cost_range)
    # This gives weight ~1 to cheapest points and decays for expensive ones.
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    
    if cost_range > 1e-9:
        # Beta controls how aggressively we penalize cost.
        # Higher beta = stronger preference for low-cost points.
        beta = 2.0
        cost_weight = np.exp(-beta * (cost_safe - cost_min) / cost_range)
    else:
        cost_weight = np.ones_like(cost_safe)
    
    # Combine PI and cost weight
    # Use geometric mean to balance: (PI * cost_weight)^(1/2)
    # Or use a weighted combination: PI^alpha * cost_weight^(1-alpha)
    # Let's use a product form which naturally balances both.
    
    scores = pi * cost_weight
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
