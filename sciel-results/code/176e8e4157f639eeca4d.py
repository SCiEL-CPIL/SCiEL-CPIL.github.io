# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
import time

# Deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds before the 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    PI(x) = Phi((f_best - mu(x)) / sigma(x))
    This favors points with a high probability of surpassing the current best,
    providing a distinct exploration bias compared to EI.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a uniform score to avoid biasing, or zeros
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    sqrt2 = np.sqrt(2.0)
    
    # Phi(z) = 0.5 * erfc(-z / sqrt2)
    # This is the Probability of Improvement
    pi = 0.5 * erfc(-z / sqrt2)
    
    # Ensure non-negative and finite
    pi = np.maximum(pi, 0.0)
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
# EVOLVE-BLOCK-END
