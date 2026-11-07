"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def _check_time():
    """Return True if we're getting close to the time limit."""
    return (time.time() - _start_time) > _TIME_LIMIT


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization."""
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (standard formula)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: use cost to the power of 0.5 to balance exploitation and cost
    # This is less aggressive than dividing by cost directly
    cost_safe = np.maximum(cost, 1e-8)
    cost_factor = np.power(cost_safe, 0.5)
    
    # Base acquisition: EI adjusted by cost
    score = ei / cost_factor
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima and explore uncertain regions
    # The bonus is small to not dominate the EI term
    exploration_bonus = 0.1 * sigma_safe
    
    score = score + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
