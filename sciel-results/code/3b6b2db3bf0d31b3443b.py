"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) Acquisition Function.
    # This is an improvement over the Quantile Acquisition Function (QAF/PI)
    # as it considers the magnitude of the expected improvement, not just the probability.
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma.
    # Higher EI means higher expected improvement.
    
    # Avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI formula for minimization:
    # Improvement is f_best - f(x).
    # E[max(f_best - f(x), 0)]
    improvement = f_best - mu
    scores = improvement * cdf_z + sigma_safe * pdf_z
    
    # Ensure scores are finite and non-negative (EI should be >= 0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
