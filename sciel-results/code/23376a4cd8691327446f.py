"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback to simple EI/cost to ensure fast return
        z = (f_best - mu) / (sigma + 1e-9)
        z = np.clip(z, -30, 30)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Deterministic Cost-Normalized Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical overflow in cdf/pdf
    z = np.clip(z, -30, 30)
    
    # Calculate Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized score: EI / cost
    # cost is (1 + mean(x)), so it's in [1, 2].
    # This directly maximizes expected improvement per unit cost.
    score = ei / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
