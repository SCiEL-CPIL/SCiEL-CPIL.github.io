import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0

def _deadline_reached():
    return time.time() - _start_time > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Cost-Adjusted Expected Improvement (EI / cost).
    
    mu: array of shape (2000,) - GP mean in objective units
    sigma: array of shape (2000,) - GP std dev in objective units
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each point, 1 + mean(x)
    
    Returns: array of shape (2000,) with acquisition scores (higher = better)
    """
    if _deadline_reached():
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # This penalizes high-cost points linearly in the denominator,
    # requiring higher expected improvement to justify the cost.
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
