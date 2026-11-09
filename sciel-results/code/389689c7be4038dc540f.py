"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 280  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Lagrangian relaxation approach where the cost constraint is handled
    by dynamically tuning a Lagrange multiplier lambda, decoupling exploration
    from the geometric distortion caused by division by cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite and safe
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Time check - if close to timeout, return a fast deterministic fallback
    if not _elapsed_ok():
        improvement = f_best - mu
        z = improvement / sigma
        ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
        ei = np.maximum(ei, 0.0)
        return ei / cost

    # 1. Calculate Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
    ei = np.maximum(ei, 0.0)
    
    # 2. Add exploration term to EI to balance exploration-exploitation
    # Use a moderate exploration weight
    alpha = 0.3
    raw_improvement = ei + alpha * sigma
    
    # 3. Lagrangian relaxation: score = raw_improvement - lambda * cost
    # We need to tune lambda so that the average cost of selected points
    # respects the budget. Since we can't track which points are selected,
    # we use a heuristic: set lambda such that points with high improvement
    # can afford higher costs.
    
    # A simple approach: normalize raw_improvement and cost to comparable scales
    raw_max = np.max(raw_improvement)
    if raw_max < 1e-9:
        raw_max = 1.0
    
    # Normalize raw_improvement to [0, 1] range for scaling
    raw_norm = raw_improvement / raw_max
    
    # Cost range is [1, 2], normalize to [0, 1]
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_range = 1.0
    cost_norm = (cost - cost_min) / cost_range
    
    # Set lambda to balance: we want the penalty to be significant but not dominant
    # If raw_improvement varies a lot, we can afford a larger lambda
    # Heuristic: lambda = 0.5 * raw_max / (cost_range + eps)
    eps = 1e-9
    lambda_val = 0.5 * raw_max / (cost_range + eps)
    
    # Clip lambda to prevent extreme values
    lambda_val = np.clip(lambda_val, 0.01, 10.0)
    
    # Compute Lagrangian scores
    scores = raw_norm - lambda_val * cost_norm
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final time check
    if not _elapsed_ok():
        return ei / cost
        
    return scores
# EVOLVE-BLOCK-END
