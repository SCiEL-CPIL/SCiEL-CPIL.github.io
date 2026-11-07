import numpy as np
from scipy.special import erf, expit

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (1-step) acquisition function.
    Mathematically equivalent to Expected Improvement (EI) for a single step.
    Uses erf for numerical stability and speed.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # CDF of standard normal: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # Using erf is often more stable for extreme values than cdf in some libraries, 
    # though scipy's cdf is robust. Let's use erf for the "new design" aspect.
    sqrt2 = np.sqrt(2.0)
    phi_z = 0.5 * (1.0 + erf(z / sqrt2))
    
    # PDF of standard normal: phi(z) = exp(-z^2 / 2) / sqrt(2 * pi)
    # To avoid underflow in exp for large z, we can calculate log-pdf, but 
    # for z > ~40, pdf is effectively 0.
    # np.exp(-0.5 * z**2) can underflow to 0. This is fine as the term vanishes.
    pdf_z = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    
    # KG (1-step) = EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    kg = (f_best - mu) * phi_z + sigma_safe * pdf_z
    
    # Ensure non-negative
    kg = np.maximum(kg, 0.0)
    
    return kg
