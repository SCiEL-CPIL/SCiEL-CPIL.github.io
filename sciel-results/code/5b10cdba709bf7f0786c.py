"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer to track elapsed time for the deadline guard
_start_time = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a combination of Probability of Improvement (POI)
    and Expected Improvement (EI) for minimization of Branin 2D.
    
    POI is used to escape local optima and explore different regions,
    while EI provides a more nuanced measure of expected improvement.
    """
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are running out of time, fall back to a very fast, simple heuristic
        # or just return the mean-based score to ensure we finish.
        # Returning -mu is the most exploitative and fast option.
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is finite and non-negative
    sigma = np.maximum(sigma, 1e-12)
    
    # Calculate Probability of Improvement (POI) for minimization
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z_poi = (f_best - mu) / sigma
    poi = norm.cdf(z_poi)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # But for minimization, we want to maximize the expected reduction
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) when f_best > mu
    # Otherwise EI = 0
    z = (f_best - mu) / sigma
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected improvement for minimization
    ei = np.where(f_best > mu, (f_best - mu) * Phi_z + sigma * phi_z, 0.0)
    
    # Combine POI and EI with a weighting
    # POI helps with exploration, EI with exploitation
    # We use a dynamic weighting that favors POI when uncertainty is high
    # and EI when uncertainty is low
    sigma_mean = np.mean(sigma)
    if sigma_mean > 0:
        # Weight POI more when there's more uncertainty
        poi_weight = 0.5 + 0.5 * np.exp(-sigma_mean)
        ei_weight = 1.0 - poi_weight
    else:
        poi_weight = 0.5
        ei_weight = 0.5
    
    # Normalize EI to be comparable to POI (which is in [0, 1])
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = ei
    
    # Combine scores
    scores = poi_weight * poi + ei_weight * ei_normalized
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
