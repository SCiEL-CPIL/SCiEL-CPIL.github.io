"""Rewritten program with Thompson Sampling acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 320.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using Thompson Sampling.
    
    mu, sigma: shape (2000,) - GP mean and std (in objective units)
    f_best: scalar - lowest observed value
    cost: shape (2000,) - evaluation cost for each candidate
    
    Returns: shape (2000,) - scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Guard against zero/negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample a single scalar theta from N(mu, sigma) for each candidate
    # This naturally balances exploration and exploitation
    # Sample theta for all 2000 candidates
    theta = mu + sigma * np.random.standard_normal(len(mu))
    
    # Score: negative cost-normalized value (lower theta is better for minimization)
    # We want higher scores for better candidates, so we negate
    # Score = -theta / cost
    score = -theta / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
