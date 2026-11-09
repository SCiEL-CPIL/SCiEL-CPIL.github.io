"""Rewritten program with cost-normalized POI + deterministic regret lower bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized probability of improvement (POI) combined with a
    deterministic regret lower bound.
    
    Formula: (f_best - mu + 0.5 * sigma) / cost
    
    This linear score balances exploitation (f_best - mu) and exploration
    (sigma) while accounting for the evaluation cost.
    """
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate the new score
    scores = (f_best - mu + 0.5 * sigma_safe) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
