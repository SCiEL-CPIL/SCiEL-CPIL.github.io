"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses Expected Improvement (EI) for minimization.
EI provides a principled balance between exploration and exploitation
by maximizing the expected reduction in the objective function value.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: Simple UCB with moderate exploration
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
    # where Z = (f_best - mu(x)) / sigma(x)
    
    # Calculate Z score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to prevent numerical underflow/overflow in pdf/cdf
    # norm.pdf and norm.cdf are stable, but clipping ensures finite results
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Compute PDF and CDF
    phi_z = norm.pdf(z_clipped)
    cdf_z = norm.cdf(z_clipped)
    
    # Ensure finite values
    phi_z = np.where(np.isfinite(phi_z), phi_z, 0.0)
    cdf_z = np.where(np.isfinite(cdf_z), cdf_z, 0.0)
    
    # Compute EI scores
    scores = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
