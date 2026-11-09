import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)

    # Ensure sigma is non-negative and finite, and avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu < f_best, z is positive.
    z = (f_best - mu) / sigma
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Using z: EI = sigma * (z * Phi(z) + phi(z))
    
    # Calculate CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = sigma * (z * Phi_z + phi_z)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
