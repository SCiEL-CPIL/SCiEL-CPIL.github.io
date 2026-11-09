# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    
    Uses Expected Improvement (EI) acquisition function.
    EI estimates the expected amount of improvement over the current best value.
    It naturally balances exploration and exploitation.
    Score = E[max(0, f_best - f(x))]
    Higher scores are preferred.
    """
    # Avoid division by zero in z-score calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal distribution
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    score = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
