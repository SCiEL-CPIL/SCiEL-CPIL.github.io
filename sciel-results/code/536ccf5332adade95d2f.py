"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
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
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware:
    # Use a mild cost penalty. The cost is 1 + mean(x_scaled) which is in [1, 2].
    # A linear penalty is reasonable, but we can also consider a slightly superlinear
    # penalty to be more cost-aware without being too aggressive.
    cost_penalized = np.maximum(cost, 1e-6)
    
    # Base score: EI / cost
    base_score = ei / cost_penalized
    
    # Add a small exploration bonus to help escape local minima.
    # This is scaled by the normalized uncertainty and the max EI.
    sigma_max = np.max(sigma_safe)
    ei_max = np.max(ei)
    if sigma_max > 1e-9 and ei_max > 0:
        exploration_bonus = 0.05 * (sigma_safe / sigma_max) * ei_max
    else:
        exploration_bonus = 0.0
    
    scores = base_score + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
