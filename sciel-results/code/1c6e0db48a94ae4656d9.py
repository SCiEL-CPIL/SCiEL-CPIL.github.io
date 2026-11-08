"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time and return fallback if too close to limit
_start_time = time.time()
_DEADLINE = 325.0  # 325s to leave 5s margin before 330s kill

def _elapsed():
    return time.time() - _start_time

def _fallback(mu, cost):
    # Fallback: minimize cost (prefer cheap evaluations)
    return -cost

def acquisition(mu, sigma, f_best, cost):
    # Check deadline - if too close, return simple fallback
    if _elapsed() > _DEADLINE:
        return _fallback(mu, cost)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost-aware acquisition: combine EI with exploration bonus
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add exploration term: sigma / cost encourages exploring uncertain cheap regions
    # Use log-expected improvement style: (EI + kappa * sigma) / cost
    # kappa controls exploration-exploitation tradeoff
    kappa = 0.5
    score = (ei + kappa * sigma_safe) / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
