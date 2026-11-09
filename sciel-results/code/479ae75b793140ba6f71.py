"""Rewritten program using pure Expected Improvement acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Expected Improvement (EI) - the canonical acquisition function
    that directly maximizes expected reduction in best observed value.
    """
    # Clip sigma to avoid division by zero and ensure stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score: how many sigma below f_best is mu
    z = (f_best - mu) / sigma
    
    # Compute CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is the expected amount by which a new sample would beat f_best
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
