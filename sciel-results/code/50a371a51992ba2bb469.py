import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: return safe scores if approaching time limit
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are at the deadline, return a neutral score (e.g., just -mu)
        # to force a selection quickly.
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    # Add a small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = E[max(0, f_best - f(x))]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Adaptive exploration parameter xi
    # Scales with the median posterior uncertainty to balance exploration/exploitation
    # A positive xi makes the target f_best + xi, which is higher (worse) than f_best,
    # thus making the EI larger for points with higher uncertainty (more exploratory).
    # If xi is negative, it is more exploitative.
    # We use a small positive value to encourage slight exploration to escape local minima.
    median_sigma = np.median(sigma_safe)
    xi = 0.1 * median_sigma
    
    f_target = f_best + xi
    
    # Z-score for the improvement
    z = (f_target - mu) / sigma_safe
    
    # Calculate EI
    # norm.cdf and norm.pdf are vectorized
    ei = (f_target - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    # If there are any NaNs or Infs, replace them with 0 or a small negative value
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
