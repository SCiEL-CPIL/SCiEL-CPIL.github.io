"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a small
    exploration term for robustness on the multimodal Styblinski-Tang 1D.
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement with small exploration parameter xi
    # xi > 0 makes EI more exploratory by reducing the improvement threshold
    xi = 0.01
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    
    # Numerically stable EI computation
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus proportional to sigma
    # This helps escape local optima in the multimodal landscape
    epsilon = 0.005
    score = ei + epsilon * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
