import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Effective Upper Confidence Bound for minimization
    # We want to minimize f(x). 
    # The "optimistic" value is mu - beta * sigma.
    # We want to select the point with the lowest optimistic value per unit cost.
    # Score = -(mu - beta * sigma) / cost
    # Higher score is better.
    
    # Adaptive beta based on f_best
    # If f_best is high, we are far from optimum, explore more (higher beta).
    # If f_best is low, we are close, exploit more (lower beta).
    # Ackley range is roughly [0, 7].
    # Map f_best to beta in [1.5, 3.0]
    
    f_best_clipped = np.clip(f_best, 0, 7.0)
    
    # Linear interpolation
    # f_best=0 -> beta=1.5
    # f_best=7 -> beta=3.0
    beta = 1.5 + (3.0 - 1.5) * (f_best_clipped / 7.0)
    
    # Ensure sigma is non-negative and handle division by zero in cost
    # cost is in [1, 2], so no zero division risk, but add epsilon for safety
    safe_cost = cost + 1e-9
    sigma = np.maximum(sigma, 1e-9)
    
    ucb = mu - beta * sigma
    scores = -ucb / safe_cost
    
    return scores
