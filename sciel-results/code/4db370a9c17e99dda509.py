import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    t_start = time.time()
    DEADLINE = t_start + 280

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF of the standard normal distribution
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    # Calculate Expected Improvement
    ei = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # Ensure all values are finite
    ei[~np.isfinite(ei)] = 0.0
    
    return ei
