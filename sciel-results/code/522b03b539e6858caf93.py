"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) as the base
    2. Add a small exploration bonus based on uncertainty (sigma)
    3. Penalize points that are too close to f_best (to encourage exploration)
    
    The key insight: for sphere 1d on [-5,5], we want to balance exploitation
    (finding low values) with exploration (covering the domain).
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small uncertainty bonus to encourage exploration
    # This helps escape local optima in the acquisition landscape
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine: EI + exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
