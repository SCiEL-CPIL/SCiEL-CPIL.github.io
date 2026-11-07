import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of goldstein price 2d.
    Uses Expected Improvement (EI) with a small exploration term.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardize for EI calculation
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Add a small exploration bonus proportional to sigma
    # This helps escape local optima while still favoring exploitation
    exploration_weight = 0.1
    score = ei + exploration_weight * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
