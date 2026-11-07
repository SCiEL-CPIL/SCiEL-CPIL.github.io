import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # 330s total limit. Let's say we want to be done by 325s.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Return a simple exploitation score: -mu
        # This ensures we pick the best current point if we are running out of time
        return -np.asarray(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive xi:
    # Early on (high f_best), we want to explore. Large xi requires significant improvement.
    # Late on (low f_best), we want to exploit. Small xi allows small improvements.
    # Branin min is ~0.4. Max is ~302.
    # If f_best is 300, xi = 10. We need to drop by 10 to get EI.
    # If f_best is 1.0, xi = 0.05.
    # If f_best is 0.4, xi = 0.02.
    
    # Scale xi with f_best
    xi = 0.05 * f_best
    
    # Calculate z
    # z = (f_best - xi - mu) / sigma
    z = (f_best - xi - mu) / sigma_safe
    
    # Calculate EI
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative EI?
    # EI can be negative if (f_best - xi - mu) is very negative and z is very negative?
    # No, if mu > f_best - xi, z is negative.
    # If mu is very high, z is very negative. cdf(z) ~ 0. pdf(z) ~ 0. EI ~ 0.
    # If mu is low, z is positive. cdf(z) ~ 1. pdf(z) ~ 0. EI ~ f_best - xi - mu.
    # EI is always non-negative for standard EI formula?
    # Actually, if we define improvement as max(0, f_best - xi - f(x)), the expectation is non-negative.
    # The formula (f_best - xi - mu)Phi(z) + sigma phi(z) is the expectation of max(0, ...).
    # So it is non-negative.
    
    # To ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
