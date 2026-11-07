"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function that balances expected improvement against cost.
    
    Uses a combination of:
    1. Expected Improvement (EI) normalized by cost
    2. A small exploration bonus based on upper confidence bound (UCB)
    
    The exploration term helps escape local optima, while the exploitation
    term focuses on improving the best found solution.
    """
    # Safety: handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardize for numerical stability
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Expected Improvement
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, 1e-8)
    ei_cost = ei / cost_safe
    
    # Upper Confidence Bound for exploration
    # Beta controls the exploration-exploitation tradeoff
    beta = 0.1
    ucb = -mu + beta * sigma_safe
    
    # Normalize ucb to be comparable in scale to ei_cost
    # Use a soft combination
    scores = ei_cost + 0.05 * ucb / (1.0 + cost_safe)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
