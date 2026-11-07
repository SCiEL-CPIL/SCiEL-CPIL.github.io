# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    # Deadline guard to ensure we don't exceed time limits
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 10:
        return np.zeros_like(mu, dtype=float)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)

    # Strategy: Expected Improvement (EI) for minimization
    # EI is often superior to fixed-beta LCB as it adapts the exploration
    # based on the potential improvement over the current best.
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # EI formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using scipy.stats.norm for numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure the output is finite to prevent invalid results
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
