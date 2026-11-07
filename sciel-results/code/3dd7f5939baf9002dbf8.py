import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted Expected Improvement (cEI).
    Higher score = more likely to be evaluated.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: exploit best mean with cost penalty
        scores = -mu / cost
        if not np.all(np.isfinite(scores)):
            scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure sigma is non-negative and has a small epsilon to avoid division by zero
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # Handle potential overflow in exp for phi, though Z is usually manageable
    # phi(Z) = exp(-0.5 * Z^2) / sqrt(2 * pi)
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)
    
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Ensure EI is non-negative (numerical errors might make it slightly negative)
    EI = np.maximum(EI, 0.0)
    
    # Cost-weighted EI
    scores = EI / cost
    
    # Safety check for finite values
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
