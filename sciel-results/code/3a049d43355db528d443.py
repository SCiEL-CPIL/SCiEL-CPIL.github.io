"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values in inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    # Clip sigma to avoid division by zero and extreme values
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Deadline guard: track elapsed time
    t_start = time.time()
    
    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB = mu - kappa * sigma
    # We want to minimize LCB, so the acquisition score should be higher for lower LCB
    # Score = -LCB / cost = (kappa * sigma - mu) / cost
    
    # Choose kappa to balance exploration and exploitation
    # A moderate kappa value is typical for LCB
    kappa = 1.0
    
    # Compute LCB-based score
    # Higher score means we should evaluate next
    # For minimization, we want low mu and high sigma (uncertainty)
    lcb_score = kappa * sigma - mu
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    score = lcb_score / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if time.time() - t_start > 290:
        # This should not happen, but just in case
        return score
        
    return score
# EVOLVE-BLOCK-END
