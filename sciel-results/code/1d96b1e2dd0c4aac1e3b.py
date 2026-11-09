import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with an exploration bonus, normalized by evaluation cost.
    """
    # Ensure sigma is positive to avoid division by zero in z-score
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu > f_best, z is negative, PI is low.
    # If mu < f_best, z is positive, PI is high.
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Exploration bonus: encourages sampling in high-uncertainty regions
    # Scaled by max sigma to keep it relative and bounded
    max_sigma = np.max(sigma)
    if max_sigma > 0:
        exploration_bonus = 0.1 * (sigma / max_sigma)
    else:
        exploration_bonus = 0.0
        
    # Combine exploitation (PI) and exploration (bonus)
    numerator = pi + exploration_bonus
    
    # Cost-awareness: divide by the cost of evaluating the point
    # Cost is expected to be >= 1.0, but we guard against zero
    cost = np.maximum(cost, 1e-9)
    
    score = numerator / cost
    
    return score
