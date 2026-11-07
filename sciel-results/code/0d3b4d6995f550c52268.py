"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # Return a safe fallback
        return -mu / (cost + 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (clip small negative values from numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: multiply EI by 1/cost to get expected improvement per unit cost
    score = ei / cost_safe
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return score
# EVOLVE-BLOCK-END
