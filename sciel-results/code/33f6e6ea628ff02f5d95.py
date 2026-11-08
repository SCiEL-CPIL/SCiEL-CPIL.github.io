"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = None

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return safe fallback if we're close to the 330s limit
    if _get_elapsed_time() > 310:
        # Fallback: prefer low mu and low cost (exploitation)
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        return -mu / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure finite and positive cost
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Ensure mu and sigma are finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    # Handle extreme z values
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI
    ei_cost = ei / cost_safe
    
    # Add a small exploration bonus for uncertain regions
    # This helps escape local minima by rewarding high sigma in cheap areas
    # The bonus is small relative to EI to not dominate
    exploration_bonus = sigma_safe / cost_safe * 0.1
    
    # Combine: EI/cost dominates, exploration bonus is auxiliary
    score = ei_cost + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
