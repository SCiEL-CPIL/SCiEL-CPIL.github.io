"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_module_start = time.time()
_DEADLINE = 280.0  # 20s safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _module_start > _DEADLINE:
        return -mu.astype(float)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Handle case where sigma is very small (numerical stability)
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute EI using stable formula
    # For large positive z, EI ≈ f_best - mu (almost certain improvement)
    # For large negative z, EI ≈ 0 (almost no improvement)
    
    # Use scipy for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure non-negative EI (should be by definition, but guard against numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
