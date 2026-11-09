"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Aware Upper Confidence Bound (UCB).
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Upper Confidence Bound for minimization
    # We want to minimize f(x). The optimistic (lower) bound is mu - beta * sigma.
    # The potential improvement over f_best is f_best - (mu - beta * sigma).
    beta = 1.5
    
    # Potential improvement
    improvement = f_best - (mu - beta * sigma_safe)
    improvement = np.maximum(improvement, 0.0)
    
    # Cost Awareness
    # Divide by cost to prefer cheaper evaluations
    score = improvement / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
