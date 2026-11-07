import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Args:
        mu: Array of shape (N,) with posterior means.
        sigma: Array of shape (N,) with posterior standard deviations.
        f_best: Scalar, the best observed function value so far.
        
    Returns:
        Array of shape (N,) with acquisition scores. Higher scores indicate
        more promising candidates.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF, phi is PDF of standard normal
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    scores = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return scores
