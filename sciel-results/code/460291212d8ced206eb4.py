"""Rewritten acquisition for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with a small exploration term.
    mu: (N,) array of GP mean predictions
    sigma: (N,) array of GP std predictions
    f_best: best observed value so far (lower is better)
    
    Returns: (N,) array of acquisition scores (higher = more promising)
    """
    # Ensure finite inputs
    sigma = np.maximum(sigma, 1e-12)
    
    # Standard EI for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip any non-finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Add a tiny exploration bonus proportional to uncertainty to encourage
    # exploring regions with high sigma, which can help escape local optima
    # in the Ackley landscape. Scale is small relative to EI values.
    exploration = 0.01 * sigma
    ei = ei + exploration
    
    # Ensure all outputs are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
