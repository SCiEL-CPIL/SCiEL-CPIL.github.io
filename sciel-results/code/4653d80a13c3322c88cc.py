"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: balance exploration (uncertainty) and exploitation (improvement).
    # Use a variant of Expected Improvement (EI) that accounts for cost.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # Also add a small exploration bonus to avoid always picking the same point
    # The cost is 1 + mean(x scaled to [0, 1]), so it's between 1 and 2
    # We want to favor points with lower cost and higher expected improvement
    
    # Use a softmax-like weighting to balance exploration and exploitation
    # The key insight: we want to maximize EI / cost, but also explore when sigma is high
    
    # A simple approach: EI / cost, but with a small epsilon to avoid zero
    scores = ei / (cost + 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
