import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    Uses Expected Improvement (EI) for minimization.
    """
    # Numerical stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Z-score: (f_best - mu) / sigma
    # For minimization, we care about how much mu is below f_best
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Return EI as the score (higher is better)
    return ei
