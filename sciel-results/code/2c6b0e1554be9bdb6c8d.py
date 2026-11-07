"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Check time limit to ensure we return within constraints
    if time.time() - _start_time > 320:
        # Fallback: pure exploitation
        return -mu

    # Adaptive EI with xi scaled to f_best magnitude
    # For minimization, EI(x) = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - xi - mu) / sigma
    # Use a small xi to allow slight exploration beyond current best
    # xi = 0.1 * abs(f_best) adapts to the scale of the objective
    xi = 0.1 * abs(f_best)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score
    z = (f_best - xi - mu) / sigma_safe
    
    # Compute Phi(z) using erfc for numerical stability
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    cdf_z = 0.5 * erfc(-z / np.sqrt(2))
    
    # Compute phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    pdf_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    
    # Calculate EI
    scores = (f_best - xi - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
