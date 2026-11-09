"""
Acquisition function for Bayesian optimization of Weierstrass 1D.
Uses Expected Improvement (EI) for minimization.
"""
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 300  # seconds

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - START_TIME > TIME_LIMIT - 5:
        # Fallback: return LCB with a conservative beta to ensure finite output
        # In a timeout scenario, we just want to return something valid quickly.
        # Pure EI is fast, so this is unlikely to trigger, but safe.
        sigma_safe = np.maximum(sigma, 1e-10)
        return mu - 2.0 * sigma_safe

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    # If f_best < mu, z is negative.
    # If f_best > mu, z is positive.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # I(x) = E[ max(f_best - f(x), 0) ]
    # For f(x) ~ N(mu, sigma^2):
    # I(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip to finite values to satisfy contract requirements
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
