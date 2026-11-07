"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timing guard
_ACQ_START = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s wall-clock limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses Expected Improvement (EI) with a dynamic exploration bonus.
    The EI is computed in standardized units, and we add a small
    uncertainty-weighted term to encourage exploration in early steps.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed objective value so far (lower is better).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be evaluated next.
    """
    # Check time budget
    elapsed = time.time() - _ACQ_START
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback: pick the point with lowest mu
        mu_arr = np.asarray(mu)
        return -mu_arr

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Standardize the improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI is non-negative; higher EI means more promising
    
    # Add a small exploration bonus based on sigma to encourage visiting uncertain regions
    # This helps in early stages when the GP is still learning
    # Scale the bonus relative to the range of EI values
    ei_range = np.max(ei) - np.min(ei)
    if ei_range > eps:
        sigma_bonus = 0.05 * (sigma_safe / np.max(sigma_safe))
    else:
        sigma_bonus = np.zeros_like(sigma_safe)
    
    scores = ei + sigma_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
