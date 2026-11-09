"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI).
    
    Args:
        mu: Mean of the GP prediction (shape 2000,)
        sigma: Standard deviation of the GP prediction (shape 2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each candidate (shape 2000,)
        
    Returns:
        scores: Acquisition scores (shape 2000,), higher is better.
    """
    # Convert to numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure f_best is a scalar
    f_best = float(f_best)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-9
    sigma_safe = sigma + eps
    cost_safe = cost + eps
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Cost-aware score: PoI per unit cost
    scores = poi / cost_safe
    
    # Ensure all scores are finite
    # If a score is NaN or Inf, replace with 0.0 (worst case)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
