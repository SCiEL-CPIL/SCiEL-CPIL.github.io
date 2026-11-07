import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu : array-like of shape (n,)
        Mean of the Gaussian process posterior.
    sigma : array-like of shape (n,)
        Standard deviation of the Gaussian process posterior.
    f_best : float
        The best (lowest) observed function value so far.
        
    Returns:
    scores : array of shape (n,)
        The EI scores. Higher values indicate better candidates.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure sigma is positive to avoid division by zero or negative std
    # GP std should theoretically be > 0, but numerical issues can occur.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If f_best < mu, z is negative, EI is small (low improvement expected)
    # If f_best > mu, z is positive, EI is larger (high improvement expected)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal distribution
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores
    # EI should be non-negative. Due to numerical precision, tiny negative values might appear.
    ei = np.maximum(ei, 0.0)
    
    # Handle any potential NaNs or Infs, though unlikely with safe sigma
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    return ei
