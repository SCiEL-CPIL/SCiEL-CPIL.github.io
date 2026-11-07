import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Expected Improvement (EI) for minimization.
    
    Parameters:
    -----------
    mu : array of shape (N,)
        Mean of the Gaussian Process posterior at N candidate points.
    sigma : array of shape (N,)
        Standard deviation of the Gaussian Process posterior at N candidate points.
    f_best : float
        The lowest observed function value so far.
        
    Returns:
    --------
    scores : array of shape (N,)
        Higher scores indicate points with higher potential for improvement.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized improvement z = (f_best - mu) / sigma
    # We want to minimize, so improvement is when f < f_best
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -50, 50)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
