"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to avoid timeout
_start_time = time.time()
_DEADLINE = 300.0  # Leave 30s margin before 330s kill


def _is_time_exceeded():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: balances expected improvement against evaluation cost.
    
    Uses cost-normalized EI with exploration bonus for high-uncertainty, low-cost regions.
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if _is_time_exceeded():
        # Fallback: prefer low-cost, high-improvement points
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = np.maximum((f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z), 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # Cost-aware: use a softer cost penalty than simple division
    # cost is in [1, 2], so log(1 + cost) gives a gentler penalty
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = np.log1p(cost_safe)  # log(1 + cost), range ~[0.69, 1.10]
    
    # Primary score: cost-normalized EI
    primary_score = ei / cost_penalty
    
    # Exploration bonus: reward high uncertainty in cheap regions
    # This helps escape local optima by exploring uncertain areas
    exploration = 0.1 * sigma_safe / cost_penalty
    
    # Combine: primarily exploitation, with exploration to avoid stagnation
    total_score = primary_score + exploration
    
    # Ensure finite values
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    # Ensure non-negative scores (higher is better)
    total_score = np.maximum(total_score, 0.0)
    
    return total_score
# EVOLVE-BLOCK-END
