"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function with a small exploration bonus.
    Returns 2000 scores; higher means evaluate next.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement: how many sigmas below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Add a small exploration component to avoid pure exploitation
    # This is a hybrid: mostly PI, but with a tiny UCB-like term
    # The exploration term is scaled down to keep PI dominant
    exploration = 0.1 * sigma_safe
    
    # Combine: primary signal is PI, secondary is exploration
    # We want higher scores to mean "evaluate next"
    # PI is in [0,1], higher is better
    # Exploration term rewards high uncertainty
    scores = pi + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
