"""Rewritten program with a stronger acquisition strategy for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) with a
    variance-aware exploration term to balance exploitation and exploration.
    
    The Beale function has a narrow global minimum, so pure EI can get stuck
    in local optima. We add a small exploration bonus proportional to sigma
    but dampened in high-certainty regions.
    """
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration term: encourage visiting high-uncertainty regions,
    # but scale it down as f_best gets closer to 0 (minimum)
    # This prevents over-exploration when we're already close to optimal
    exploration_weight = 0.1 * np.exp(-np.abs(f_best) * 0.5)
    exploration = exploration_weight * sigma_safe
    
    # Combine: EI dominates, exploration breaks ties in uncertain regions
    score = ei + exploration
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
