import numpy as np
import time
from scipy.stats import norm

# Global state for deadline tracking
_start_time = time.time()
_DEADLINE_LIMIT = 290.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best):
    # Deadline guard
    current_time = time.time()
    if current_time - _start_time > _DEADLINE_LIMIT:
        # Fallback: Greedy exploitation to ensure quick return
        return -mu

    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # EI is the expected improvement over the current best value f_best
    # Higher EI means higher probability of finding a better solution
    
    # Calculate Z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
