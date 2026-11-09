"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB.
    
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
    
    # UCB for minimization:
    # We want to find points where f(x) could be lower than current best.
    # The optimistic lower bound is mu - beta * sigma.
    # We maximize the potential improvement: f_best - (mu - beta * sigma)
    # = f_best - mu + beta * sigma
    
    # Tuning beta: In high dimensions with few samples, exploration is key.
    # A beta of 1.5-2.0 is often effective.
    beta = 1.5
    
    ucb_score = f_best - mu + beta * sigma_safe
    
    # Clip to non-negative to avoid selecting points that are clearly worse
    # even with exploration bonus, though the division by cost handles relative ranking.
    # However, negative scores should be penalized.
    
    # Cost-aware: divide by cost^1.5
    # Cost is in [1, 2]. cost^1.5 ranges [1, 2.82].
    cost_penalized = np.maximum(cost, 1e-6) ** 1.5
    
    scores = ucb_score / cost_penalized
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores


def cost_safe(cost):
    """Ensure cost values are safe for division."""
    return np.maximum(cost, 1e-6)
# EVOLVE-BLOCK-END
