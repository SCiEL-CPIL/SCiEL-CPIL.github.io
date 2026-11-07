"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) to balance exploration and exploitation.
    This is a structural change from UCB, using the posterior distribution
    to compute the expected gain over the current best.
    
    Parameters:
    mu (np.ndarray): Posterior mean, shape (2000,).
    sigma (np.ndarray): Posterior standard deviation, shape (2000,).
    f_best (float): Best observed value so far.
    
    Returns:
    np.ndarray: Scores for the 2000 candidates. Higher is better.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Fallback to pure exploitation to save time
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e6)
        return scores

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)

    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu_safe) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is norm.cdf, phi is norm.pdf
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu_safe) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
