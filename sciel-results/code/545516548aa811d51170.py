"""Cost-aware acquisition for Bayesian optimization of Styblinski-Tang 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _styblinski_tang_2d(x):
    """Styblinski-Tang function in 2D. x is (n, 2) in [-5, 5]^2."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x, axis=1)


def _cost_of(x):
    """Cost of evaluating x: 1 + mean(x scaled to [0, 1])."""
    x = np.asarray(x, dtype=np.float64)
    # x is in [-5, 5], scale to [0, 1]
    x_scaled = (x + 5.0) / 10.0
    return 1.0 + np.mean(x_scaled, axis=1)


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: UCB with cost normalization and structure-aware bonus.
    
    mu, sigma, cost: arrays of shape (2000,)
    Returns: array of shape (2000,) with higher = better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Deadline guard
    t0 = time.time()
    
    # UCB acquisition: -mu + beta * sigma
    # Use a beta that balances exploration and exploitation
    # Normalize sigma to avoid issues with very small/large values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic beta: start high for exploration, decrease as f_best improves
    # f_best is the lowest observed value; the true minimum is around -78.33
    # Use a heuristic: beta decreases as we get closer to the known minimum
    # But we don't know which instance, so use a fixed reasonable beta
    # Based on the observed improvement from EI to UCB, use beta ~ 2-4
    
    # Let's use a beta that adapts based on the spread of sigma
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # UCB score: -mu + beta * sigma / cost
    # We want to maximize this
    # Use beta that gives good exploration
    beta = 2.5  # Fixed beta for UCB
    
    # Base UCB score
    ucb_score = -mu + beta * sigma_safe
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    score = ucb_score / cost_safe
    
    # Structure-aware bonus: The Styblinski-Tang function has its global minimum
    # at a specific point. In the original coordinates [-5,5]^2, the minimum is at
    # approximately (-2.9035, -2.9035) for each coordinate, but since each instance
    # shifts the minimum to a uniform point in the inner 80% of each range,
    # we can't hardcode the exact location.
    # 
    # However, we can use the GP's mu and sigma to guide exploration.
    # The key insight: candidates where mu is low (good predicted value) and 
    # sigma is moderate should be prioritized.
    #
    # Add a small exploration bonus for high-sigma regions to encourage
    # exploring unexplored areas
    sigma_normalized = sigma_safe / (sigma_max + 1e-9)
    score += 0.5 * sigma_normalized
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline check
    if time.time() - t0 > 280:
        # Return the best we have
        return score
    
    return score
# EVOLVE-BLOCK-END
