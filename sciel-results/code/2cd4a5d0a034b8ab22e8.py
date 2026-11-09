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
        # Fallback: return LCB with a conservative beta
        return mu - 2.0 * sigma

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
    
    # Clip to finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply a multiplicative boost to candidates with high uncertainty
    # This encourages exploration in areas where the model is less certain
    # The boost is proportional to sigma, but capped to prevent extreme values
    # We use a soft cap to avoid numerical issues
    sigma_normalized = sigma_safe / (np.max(sigma_safe) + 1e-10)
    
    # The boost factor increases with uncertainty
    # Using a quadratic term to emphasize high-uncertainty regions
    boost = 1.0 + 0.5 * sigma_normalized
    
    # Apply the boost to the EI values
    ei_boosted = ei * boost
    
    # Ensure the result is finite
    ei_boosted = np.nan_to_num(ei_boosted, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei_boosted
