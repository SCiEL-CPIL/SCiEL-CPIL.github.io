"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    POI-based acquisition function for Bayesian optimization.
    
    Uses the Probability of Improvement explicitly scaled by the expected
    gain magnitude, with a small exploration bonus to avoid premature
    convergence.
    
    Score = POI * max(0, f_best - mu) + 0.1 * sigma
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Expected gain magnitude (clipped to be non-negative)
    expected_gain = np.maximum(f_best - mu, 0.0)
    
    # Core score: POI scaled by expected gain
    core_score = poi * expected_gain
    
    # Small exploration bonus proportional to uncertainty
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine
    score = core_score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
