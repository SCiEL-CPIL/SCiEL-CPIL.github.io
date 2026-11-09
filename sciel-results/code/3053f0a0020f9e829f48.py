import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Calculate z-score
    # If sigma is very small, z can be very large/small
    z = (f_best - mu) / sigma_safe
    
    # Clip z to [-10, 10] to prevent numerical issues and stabilize scores
    z = np.clip(z, -10, 10)
    
    # Compute CDF and PDF of standard normal
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Calculate EI scores
    scores = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
