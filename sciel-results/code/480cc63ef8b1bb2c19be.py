import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # Handle small sigma to avoid division by zero and numerical instability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Phi is CDF, phi is PDF
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # If sigma was effectively zero, EI should be max(0, f_best - mu)
    # The formula above approximates this well if z is large positive (mu < f_best)
    # If mu > f_best, z is large negative, cdf ~ 0, pdf ~ 0, ei ~ 0.
    # So the formula is robust.
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
