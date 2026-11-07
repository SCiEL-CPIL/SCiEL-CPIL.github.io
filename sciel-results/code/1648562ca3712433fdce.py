import numpy as np
from scipy.stats import t

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Degrees of freedom for the t-distribution
    nu = 5.0
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # CDF of the t-distribution
    cdf = t.cdf(z, df=nu)
    
    # PDF of the t-distribution
    pdf = t.pdf(z, df=nu)
    
    # Expected Improvement for t-distribution
    # EI = (f_best - mu) * F_nu(z) + sigma * (nu / (nu - 1)) * (1 + z^2 / nu) * t_nu(z)
    factor = (nu / (nu - 1.0)) * (1.0 + (z ** 2) / nu)
    
    ei = (f_best - mu) * cdf + sigma * factor * pdf
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
