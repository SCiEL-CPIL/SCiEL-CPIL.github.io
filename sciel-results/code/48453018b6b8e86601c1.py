"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses Expected Improvement (EI) as the acquisition function.
EI is a structural alternative to UCB that directly optimizes the
probability and magnitude of improvement over the current best,
naturally balancing exploration and exploitation without arbitrary
schedules or hyperparameters like kappa.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points using Expected Improvement.
    
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
        # Fallback: Simple EI calculation is fast, but if we are close to deadline,
        # we can just return a simple UCB-like score to be safe, though EI is vectorized.
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    # Clip sigma to avoid division by zero in the z-score calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive (improvement likely)
    # If mu > f_best, z is negative (improvement unlikely, but possible due to sigma)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # Use scipy.stats.norm for numerical stability
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    scores = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure all scores are finite and non-negative (EI should be >= 0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
