import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    
    Where Phi is the standard normal CDF and phi is the standard normal PDF.
    """
    # Handle sigma close to 0 to avoid numerical issues
    # If sigma is very small, EI is approximately max(0, f_best - mu)
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # EI formula
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
