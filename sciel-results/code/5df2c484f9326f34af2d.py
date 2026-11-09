"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses Expected Improvement (EI) for minimization with numerical stability.
    EI(x) = E[max(f_best - f(x), 0)]
    For a GP posterior with mean mu and std sigma:
    z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute the standard normal score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # Use scipy.special for efficiency
    from scipy.special import ndtr, ndtri
    # ndtr is the CDF of standard normal, phi(z) = exp(-0.5*z^2) / sqrt(2*pi)
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    ei = (f_best - mu) * ndtr(z) + sigma_safe * phi_z
    
    # Ensure all scores are finite (numerical safety)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
