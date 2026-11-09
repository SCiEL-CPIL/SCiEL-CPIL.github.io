import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is not zero to avoid division by zero
    # Use a slightly larger floor to be robust
    sigma_safe = np.maximum(sigma, 1e-6)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values where pdf/cdf might lose precision
    # or behave unexpectedly in some lib versions
    z_clipped = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    # Standard EI
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    return ei
