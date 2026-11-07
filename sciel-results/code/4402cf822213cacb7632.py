"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None
_DEADLINE = 310.0  # 330s limit, leave 20s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - lowest observed value
    cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're too close to the limit, return a quick fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: simple UCB-like score
        sigma_safe = np.maximum(sigma, 1e-8)
        fallback = (f_best - mu - 1.0 * sigma_safe) / np.maximum(cost, 1e-8)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Safe cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware EI: EI per unit cost
    ei_cost = ei / cost_safe
    
    # Exploration component: UCB-like term
    # This encourages exploration of uncertain regions
    ucb = (f_best - mu - 1.0 * sigma_safe) / cost_safe
    
    # Min-Max Normalization for robust combination
    # This scales each component to [0, 1] based on the current batch of candidates
    def min_max_norm(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        if arr_range < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr_min) / arr_range

    ei_cost_norm = min_max_norm(ei_cost)
    ucb_cost_norm = min_max_norm(ucb)
    
    # Combine with a weighted sum. 
    # Increasing exploitation weight slightly to 0.8 as the GP becomes more informative.
    scores = 0.8 * ei_cost_norm + 0.2 * ucb_cost_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
