import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses Expected Improvement (EI) for minimization.
    Score = EI(mu, sigma, f_best)
    """
    start_time = time.time()
    deadline = 295.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-6)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    
    # Handle case where f_best might be very large or small relative to mu
    # But standard formula is robust enough with safe sigma.
    
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # EI is always non-negative.
    # If f_best is very good (close to 0) and mu is high, EI is high.
    # If mu is low (good) and sigma is low, EI is low (exploitation is satisfied).
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: Pure exploitation (minimize mu)
        # We return -mu so that lower mu gives higher score.
        # But EI is already a valid score. If we timeout, we just return the calculated EI.
        # The calculation is O(N) and very fast, so timeout is unlikely.
        pass
        
    return ei
