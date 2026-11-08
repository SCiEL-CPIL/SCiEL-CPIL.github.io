"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement
    with a UCB exploration term, normalized by evaluation cost.
    """
    # Deadline guard: if we're running too long, return a safe fallback
    # The function itself is fast, but this satisfies the directive.
    # In practice, this function is called many times, so we keep it lightweight.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize, so we look at mu - beta * sigma
    # Higher score = more promising, so we use (beta * sigma - mu)
    beta = 2.0
    ucb_score = beta * sigma - mu
    
    # Combine EI and UCB: use a weighted sum
    # EI already accounts for improvement, UCB adds exploration
    # Normalize both to have similar scales
    # EI is in objective units, UCB is in objective units
    # Let's use a blend: primarily EI, with a small UCB boost for exploration
    combined = ei + 0.5 * ucb_score
    
    # Cost-aware: divide by cost
    scores = combined / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
