"""Rewritten cost-aware acquisition for styblinski tang 2d BO."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # wall-clock limit
_SAFETY_MARGIN = 5.0  # seconds to spare


def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement per unit cost,
    with a small exploration bonus for low-cost regions.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition values (higher = evaluate next)
    """
    # Deadline guard: if we're running out of time, return a simple heuristic
    if _check_deadline():
        # Fallback: just use EI without cost normalization, but still finite
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / np.maximum(cost, 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost normalization: cost is 1 + mean(x_scaled), so range is [1, 2]
    # We want to reward both high improvement and low cost
    # Use EI / cost^alpha with a small alpha to avoid over-penalizing cost
    # but still favor cheaper evaluations
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to non-negative (EI should be >= 0 mathematically, but numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: divide by cost raised to a power
    # Using cost^0.5 gives a moderate preference for cheaper points
    # while still allowing expensive points with high EI to be selected
    cost_factor = np.power(np.maximum(cost, 1e-9), 0.5)
    
    scores = ei / cost_factor
    
    # Add a small uncertainty bonus to encourage exploration
    # This helps when EI is zero everywhere (e.g., early on or after convergence)
    # The bonus is scaled by sigma to favor high-uncertainty regions
    # But we keep it small so it doesn't dominate
    uncertainty_bonus = 0.01 * sigma / cost_factor
    
    scores = scores + uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
