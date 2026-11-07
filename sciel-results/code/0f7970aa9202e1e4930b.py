import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Expected Hypervolume Improvement (EHI).
    In single-objective minimization, the expected increase in the hypervolume
    dominated by the set of evaluated points (relative to the current best)
    is mathematically equivalent to the Expected Improvement (EI).
    
    EI(x) = E[max(f_best - f(x), 0)]
    = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    # If sigma is very small and mu > f_best, EI should be 0.
    # The formula handles this naturally as z -> -inf, Phi(z) -> 0, phi(z) -> 0.
    # However, if mu is very close to f_best, numerical issues might arise.
    
    # Clip to 0 to ensure non-negative scores (EI is always >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Replace any non-finite values with 0
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
