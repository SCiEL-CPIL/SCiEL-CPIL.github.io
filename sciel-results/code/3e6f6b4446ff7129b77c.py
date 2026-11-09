import numpy as np
from scipy.stats import t as student_t


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Student-t Process Expected Improvement
    with an adaptive exploration bonus.
    
    Combines the heavy-tailed predictive distribution of a Student-t Process
    (for robustness against outliers and better exploration) with an
    adaptive exploration bonus that decays as the best value improves.
    
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
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Sanitize sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.nan_to_num(sigma_safe, nan=1e-12, posinf=1e6, neginf=1e-12)
    
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
    z_t = np.clip(z_t, -10, 10)  # Numerical stability
    
    cdf_t = student_t.cdf(z_t, nu)
    pdf_t = student_t.pdf(z_t, nu)
    
    # EI for t-distribution:
    # EI = (f_best - mu) * CDF_t(z) + t_scale * nu/(nu-1) * PDF_t(z) * (1 + z^2/nu)
    correction = nu / (nu - 1.0)
    ei_t = (f_best - mu) * cdf_t + t_scale * correction * pdf_t * (1.0 + z_t**2 / nu)
    
    # Clip to avoid numerical issues
    ei_t = np.clip(ei_t, -1e6, 1e6)
    
    # Adaptive exploration bonus
    # beta decays as f_best decreases, encouraging more exploitation later
    # f_best is typically in range [0, ~100] for Beale function
    # When f_best is large (early), beta is large (more exploration)
    # When f_best is small (late), beta is small (more exploitation)
    beta = 0.1 + 0.5 * (f_best / (f_best + 3.0))
    
    # Combine EI with exploration bonus
    scores = ei_t + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
