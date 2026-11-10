"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Return a deterministic fallback if we're over deadline
        # Use cost-aware EI as fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.clip(sigma, 1e-9, None)
    cost_safe = np.clip(cost, 1e-9, None)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Exploration term: PI * sigma represents expected improvement from exploring
    # uncertain regions where there's a high probability of finding a better solution
    exploration_term = 0.5 * pi * sigma_safe
    
    # Combine EI and exploration term
    acquisition_value = ei + exploration_term
    
    # Cost normalization: normalize costs to [0, 1] range for better scaling
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.ones_like(cost_safe)
    else:
        cost_norm = (cost_safe - cost_min) / cost_range
    
    # Invert cost norm so lower cost gets higher weight
    cost_weight = 1.0 / (cost_norm + 0.1)  # Add small epsilon for stability
    
    # Normalize acquisition value to comparable scale
    acq_max = np.max(acquisition_value)
    if acq_max < 1e-12:
        acq_norm = np.zeros_like(acquisition_value)
    else:
        acq_norm = acquisition_value / acq_max
    
    # Combined acquisition score
    score = acq_norm * cost_weight
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
