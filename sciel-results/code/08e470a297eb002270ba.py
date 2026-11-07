import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses Expected Improvement (EI) for minimization.
    
    Parameters:
    mu (np.array): Posterior mean, shape (2000,)
    sigma (np.array): Posterior standard deviation, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the candidates, shape (2000,). Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for Expected Improvement
    # For minimization, improvement is defined as f_best - f(x)
    # We maximize the expected improvement.
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is the CDF of the standard normal distribution
    # phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    return ei
