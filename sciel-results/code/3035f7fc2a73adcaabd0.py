"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    For minimization, EI is defined as the expected value of the improvement
    over the current best f_best.
    EI(x) = E[max(f_best - f(x), 0)]
    
    Given f(x) ~ N(mu, sigma^2):
    z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    We return these scores, so higher scores are preferred.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
