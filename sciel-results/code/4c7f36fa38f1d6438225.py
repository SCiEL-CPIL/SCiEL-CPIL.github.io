"""Rewritten program using a SEARCH approach within the time budget."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses a hybrid EI + PI acquisition with weighted combination favoring exploitation.
    """
    # Clip sigma to avoid division by zero and ensure stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Compute CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Probability of Improvement: Phi(z)
    pi = cdf
    
    # Normalize EI and PI to comparable scales
    # EI is in objective units, PI is in [0, 1]
    # Scale PI by the scale of EI to make them comparable
    # Use a robust scale estimate
    ei_scale = np.max(ei) - np.min(ei) + 1e-10
    
    # Weight: 70% PI, 30% EI (PI favors exploitation)
    w_pi = 0.7
    w_ei = 0.3
    
    # Combine: scale PI to EI's scale, then weighted sum
    hybrid = w_ei * ei + w_pi * (pi * ei_scale)
    
    # Ensure all scores are finite
    hybrid = np.where(np.isfinite(hybrid), hybrid, 0.0)
    
    return hybrid
# EVOLVE-BLOCK-END
