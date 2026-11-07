"""Rewritten acquisition function for cost-aware Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement per unit cost.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative cost
    cost = np.where(cost > 1e-12, cost, 1.0)
    sigma = np.where(sigma > 1e-12, sigma, 1e-9)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
