"""Rewritten program using UCB acquisition with refined adaptive beta and stronger sigma penalty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Upper Confidence Bound (UCB) with refined adaptive beta parameter 
    and a stronger penalty on high uncertainty to focus search.
    """
    # Clip sigma to avoid division by zero and ensure stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Normalize f_best to estimate progress
    # f_best starts around 25 (5^2) and approaches 0
    f_norm = np.clip(f_best / 25.0, 0, 1)
    
    # Adaptive beta: 
    # Start with high exploration (~2.5), decrease to low exploitation (~0.4).
    # Use a power of 1.2 for a smoother transition than linear or 1.5.
    beta = 0.4 + 2.1 * (f_norm ** 1.2)
    
    # UCB score: higher means more likely to be evaluated next
    # For minimization: we want low mu and high sigma (uncertainty)
    ucb_score = -mu + beta * sigma
    
    # Add a penalty for high sigma to prevent pure exploration in far regions
    # This helps keep the search focused on the basin of attraction.
    # Increased penalty strength (0.05) and lower threshold (0.5) to be more effective.
    sigma_penalty = 0.05 * np.maximum(sigma - 0.5, 0) ** 2
    ucb_score = ucb_score - sigma_penalty
    
    # Ensure all scores are finite
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
# EVOLVE-BLOCK-END
