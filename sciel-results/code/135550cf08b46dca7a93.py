"""Rewritten program: POI-scaled acquisition with exploration bonus for ST1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability-of-Improvement based acquisition with exploration scaling.
    
    For each candidate:
      - z = (f_best - mu) / sigma  (positive when mu < f_best, i.e., promising)
      - POI = Phi(z)              (probability that f(x) < f_best)
      - Expected gain magnitude = (f_best - mu) when mu < f_best, else 0
      - Score = POI * (|f_best - mu| + sigma)  -- blends exploitation with exploration
    
    This is structurally different from pure EI: it weights the probability of
    improvement by a gain magnitude that includes the uncertainty (sigma),
    which encourages visiting high-variance regions where a large improvement
    is possible, not just where the mean looks best.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Gain magnitude: how much better could we get?
    # Use |f_best - mu| clipped to non-negative, plus sigma for exploration
    gain_magnitude = np.maximum(f_best - mu, 0.0) + sigma_safe
    
    # Score: POI weighted by gain magnitude
    # Higher score = more likely to improve AND larger potential gain
    scores = poi * gain_magnitude
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
