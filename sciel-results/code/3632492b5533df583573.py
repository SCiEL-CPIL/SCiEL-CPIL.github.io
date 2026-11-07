import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Use UCB (Upper Confidence Bound) for minimization.
    # Score is high for points with low predicted value and high uncertainty.
    # Penalize by cost to prefer cheaper evaluations.
    
    # Beta controls exploration vs exploitation.
    # With low noise, a moderate beta is appropriate.
    beta = 2.0
    
    # Lower Confidence Bound (potential minimum)
    lcb = mu - beta * sigma
    
    # We want to minimize lcb, so we maximize -lcb.
    # Divide by cost to account for evaluation expense.
    score = -lcb / cost
    
    # Ensure output is finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
