import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes Expected Improvement (EI) scores for Bayesian optimization.
    
    Args:
        mu: numpy array of shape (N,) representing the posterior mean.
        sigma: numpy array of shape (N,) representing the posterior standard deviation.
        f_best: float, the best (lowest) value observed so far.
        
    Returns:
        numpy array of shape (N,) with EI scores. Higher is better.
    """
    # Ensure sigma is strictly positive to avoid division by zero or log issues
    # Although the problem states sigma is a standard deviation, numerical issues can occur
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute the improvement: f_best - mu
    # We are minimizing, so improvement is positive if mu < f_best
    improvement = f_best - mu
    
    # Standardize the improvement
    # z = (f_best - mu) / sigma
    z = improvement / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF of the standard normal distribution
    # And phi is the PDF of the standard normal distribution
    
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure output is finite
    # If any non-finite values arise (e.g., from extreme inputs), replace with 0 or a small negative
    # In practice, with safe sigma, this should be stable.
    # norm.cdf and norm.pdf are well-behaved for finite inputs.
    # However, if mu or sigma were infinite, z could be NaN.
    # The problem states mu and sigma are in objective units and likely finite.
    # But let's be robust.
    if not np.all(np.isfinite(ei)):
        ei = np.where(np.isfinite(ei), ei, 0.0)
        
    return ei
