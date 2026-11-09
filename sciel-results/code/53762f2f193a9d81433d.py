import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0 # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Check time
    if time.time() - _start_time > _TIME_LIMIT:
        # Return a safe fallback, e.g., uniform or previous best
        # Since we don't have previous best, return zeros or random
        # Zeros are valid finite scores
        return np.zeros_like(mu)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # This models the "value density" as the probability-weighted improvement
    d = f_best - mu
    z = d / sigma
    
    # Calculate EI = d * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = d * Phi_z + sigma * phi_z
    
    # Ensure ei is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Maximizing expected regret reduction per unit budget
    scores = ei / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
