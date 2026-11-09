"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Student-t Process (STP) Expected Improvement.
    
    Uses a scaled t-distribution (3 degrees of freedom) to approximate the
    predictive distribution, yielding heavier tails for uncertainty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Student-t Process with nu = 3 degrees of freedom
    # Scale the t-distribution to match the GP posterior variance
    # For a t-distribution with nu df, Var = nu/(nu-2) * scale^2
    # So scale = sigma * sqrt((nu-2)/nu) to match variance
    nu = 3.0
    scale_factor = np.sqrt((nu - 2.0) / nu)
    t_scale = sigma_safe * scale_factor
    
    # Expected Improvement for minimization with Student-t predictive
    # For t-distribution: P(f < f_best) = CDF_t((f_best - mu) / t_scale)
    # EI = E[(f_best - f)^+ | f ~ t(mu, t_scale, nu)]
    
    z_t = (f_best - mu) / t_scale
    
    # CDF and PDF of Student-t
    cdf_t = student_t.cdf(z_t, nu)
    pdf_t = student_t.pdf(z_t, nu)
    
    # EI for t-distribution:
    # EI = (f_best - mu) * CDF_t(z) + t_scale * nu/(nu-1) * PDF_t(z) * (1 + z^2/nu)
    # This comes from the expectation of max(0, f_best - f) where f ~ t(mu, scale, nu)
    
    correction = nu / (nu - 1.0)
    ei_t = (f_best - mu) * cdf_t + t_scale * correction * pdf_t * (1.0 + z_t**2 / nu)
    
    # Clip to avoid numerical issues
    ei_t = np.clip(ei_t, -1e6, 1e6)
    
    # Add a small exploration bonus
    scores = ei_t + 0.01 * sigma_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
