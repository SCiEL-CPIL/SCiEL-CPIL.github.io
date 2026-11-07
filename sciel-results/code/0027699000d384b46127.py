import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Log-Expected Improvement (Log-EI) acquisition function for minimization.
    
    Parameters:
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior standard deviation
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (2000,), acquisition values (higher is better)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # where f(x) ~ N(mu, sigma^2)
    # Let Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is always non-negative.
    # Use log(1 + EI) to compress large values and handle cases where EI is 0.
    # This is a common robust variant of EI.
    scores = np.log1p(ei)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
