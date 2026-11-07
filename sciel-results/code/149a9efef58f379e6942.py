"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Variance-Gated Expected Improvement (VEI) for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI for minimization
    ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Variance-Gated Expected Improvement (VEI)
    # Scale EI by (sigma / sigma_max)^gamma
    # This penalizes exploitation in low-uncertainty regions
    # and encourages exploration in high-uncertainty regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > eps:
        gate = (sigma_safe / sigma_max) ** 1.5
    else:
        gate = np.ones_like(sigma_safe)
        
    scores = ei_scores * gate
    
    # Ensure all scores are finite and non-negative
    scores = np.maximum(scores, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
