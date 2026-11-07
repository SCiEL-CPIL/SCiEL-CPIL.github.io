"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Uses sqrt(cost) normalization to reduce the penalty for expensive evaluations,
    allowing better exploration of high-cost regions that may contain the global minimum.
    """
    # Ensure sigma is positive to avoid division by zero issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware normalization
    # Using sqrt(cost) instead of cost to soften the penalty for high-cost points
    cost_safe = np.maximum(cost, 1e-8)
    cost_factor = np.sqrt(cost_safe)
    
    # Return the score (higher is better)
    scores = ei / cost_factor
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
