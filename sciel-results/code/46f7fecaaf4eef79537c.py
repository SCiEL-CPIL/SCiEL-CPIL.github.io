import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    
    This is a structural change from the previous UCB-based approach.
    EI directly optimizes the expected improvement over the current best value f_best.
    It naturally balances exploration and exploitation without requiring a 
    user-defined exploration parameter (beta).
    
    For minimization:
    EI(x) = E[max(f_best - f(x), 0)]
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite to avoid division by zero or NaNs
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Handle non-finite f_best
    if not np.isfinite(f_best):
        f_best = 1e6
        
    # Calculate z-score for the improvement
    # z = (f_best - mu) / sigma
    # If f_best is much larger than mu, z is large positive -> high probability of improvement
    # If f_best is much smaller than mu, z is large negative -> low probability
    
    # Clip z to prevent numerical overflow in exp/pdf/cdf
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    # Expected Improvement formula for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Ensure finite output
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    return ei
