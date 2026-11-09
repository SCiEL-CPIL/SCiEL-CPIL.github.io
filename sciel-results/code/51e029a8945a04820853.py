import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Adjusted Probability of Improvement (PoI) with an uncertainty bonus.
    Score = (P(improvement) + lambda * sigma) / cost
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for Probability of Improvement (minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    
    # Probability of Improvement: P(mu < f_best)
    poi = norm.cdf(z)
    
    # Exploration bonus
    # We want to balance the probability (0 to 1) with the uncertainty (sigma).
    # A lambda of 0.5 is a reasonable starting point to allow exploration
    # without completely ignoring the improvement likelihood.
    lambda_explore = 0.5
    bonus = lambda_explore * sigma_safe
    
    # Combined numerator
    numerator = poi + bonus
    
    # Cost adjustment
    # Higher cost reduces the score.
    scores = numerator / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
