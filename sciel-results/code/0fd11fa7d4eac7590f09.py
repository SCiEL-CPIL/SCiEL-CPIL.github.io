"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (EI) with additive cost penalty.
    
    Strategy:
    1. Compute standard Expected Improvement (EI) from the GP posterior.
       EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma.
       This is a robust, standard acquisition function that balances exploitation 
       (low mu) and exploration (high sigma) in a probabilistically justified way.
       
    2. Apply an additive cost penalty: score = EI - lambda * cost.
       This creates a linear trade-off between the expected improvement and the 
       evaluation cost. Unlike dividing by cost (multiplicative), this avoids 
       over-penalizing regions with moderate cost but high EI, and avoids 
       under-penalizing very cheap regions if EI is low.
       
    3. Use a small lambda (0.1) to gently guide the search toward cheaper regions 
       without dominating the EI signal. The minimizer is in the inner 80% box, 
       so cost varies between 1.4 and 1.6 roughly; a small penalty helps break 
       ties in favor of cheaper points without preventing exploration of the 
       entire inner box.
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple exploitation
        scores = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For numerical stability, we can use the fact that for large negative z, EI ~ 0
    # and for large positive z, EI ~ (f_best - mu)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip EI to be non-negative (should be by definition, but for numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Additive cost penalty with small lambda
    lambda_cost = 0.1
    scores = ei - lambda_cost * cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
