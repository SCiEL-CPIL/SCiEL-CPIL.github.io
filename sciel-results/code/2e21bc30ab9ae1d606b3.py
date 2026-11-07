"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Safety margin: return a simple fallback if we're too close to the deadline
    if time.time() - start_time > 295.0:
        return -mu / cost
    
    # Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    # For a GP with mean mu and std sigma:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: use a global z-score penalty instead of point-wise division
    # This better balances exploration-exploitation by penalizing expensive candidates
    # relative to the batch mean rather than locally.
    
    # Compute cost statistics
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-9
    
    # Z-score of cost (higher cost = more negative z-score)
    cost_z = (cost - cost_mean) / cost_std
    
    # Apply cost penalty: subtract a scaled version of cost_z from the EI
    # Candidates with higher-than-average cost get penalized more
    # Use a tunable penalty strength
    penalty_strength = 0.5
    
    scores = ei - penalty_strength * cost_z * np.std(ei + 1e-9)
    
    # Alternative: multiplicative cost-aware scaling
    # Scale EI by a factor that decreases with cost
    # cost_factor = 1.0 / (1.0 + 0.3 * np.maximum(cost_z, 0))
    # scores = ei * cost_factor
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Time guard
    if time.time() - start_time > 295.0:
        return -mu / cost
    
    return scores
# EVOLVE-BLOCK-END
