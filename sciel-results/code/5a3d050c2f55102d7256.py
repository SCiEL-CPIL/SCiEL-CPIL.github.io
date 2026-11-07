"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0  # Leave 30s buffer before 330s kill


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running low on time, return a simple score
    if _remaining_time() < 2.0:
        # Fallback: use simple EI without cost normalization
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Cost-aware expected improvement with exploration bonus
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: divide by cost, but add small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    cost_aware_ei = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty to avoid premature convergence
    # This helps when f_best is already very good and EI becomes tiny
    exploration_bonus = 0.01 * sigma_safe / cost_safe
    
    # Combine: use cost-aware EI with a floor to maintain exploration
    score = cost_aware_ei + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
