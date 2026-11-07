"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Probability of Improvement (POI) divided by evaluation cost.
    
    POI measures the probability that the objective at x is better than f_best,
    normalized by cost to prefer cheaper candidates.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for probability of improvement
    # z = (f_best - mu) / sigma
    # POI = P(f(x) < f_best) = Phi(z)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Normalize by cost: prefer cheaper evaluations
    # cost is in range [1, 2] (1 + mean of x scaled to [0,1])
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score: POI / cost (higher is better)
    score = poi / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
