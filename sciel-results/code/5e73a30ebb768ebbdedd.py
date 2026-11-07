"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    Z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi(Z) = norm.cdf(Z)
    # phi(Z) = norm.pdf(Z)
    scores = (f_best - mu) * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
