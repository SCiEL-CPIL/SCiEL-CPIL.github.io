"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
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
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is CDF of standard normal, phi is PDF
    
    # Compute the standardized improvement
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute Phi(z) and phi(z)
    from scipy.special import ndtr, ndtri
    
    # Phi(z) = ndtr(z)
    phi_z = ndtr(z)
    
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    scores = (f_best - mu) * phi_z + sigma * pdf_z
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Ensure non-negative scores
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
