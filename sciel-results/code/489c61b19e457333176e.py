import numpy as np
from scipy.stats import norm
import time

# Track elapsed time from module load for the deadline guard
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses Probability of Improvement (PI) as the primary signal,
    which aggressively exploits the current best region.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    For minimization, we want to find points where the probability
    of improving upon f_best is highest.
    """
    # Deadline guard: if we're running out of time, return a safe default
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Emergency fallback: return zeros (uniform exploration)
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Probability of Improvement (PI) for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in CDF computation
    z = np.clip(z, -50, 50)
    
    # PI is the CDF of the standard normal at z
    pi_scores = norm.cdf(z)
    
    # Ensure scores are finite
    pi_scores = np.nan_to_num(pi_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return pi_scores
