import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Styblinski-Tang 2D.
    
    Uses Upper Confidence Bound (UCB) with a linear cost penalty.
    UCB encourages exploration of high-uncertainty regions while exploiting low-mean regions.
    The linear cost penalty favors the low-cost quadrant where the global minimum is located.
    """
    # Ensure sigma and cost are positive and finite
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # UCB for minimization:
    # We want to maximize the potential for finding a lower value.
    # A point is attractive if it has a low mean (mu) or high uncertainty (sigma).
    # Score component: beta * sigma - mu
    beta = 2.5
    
    ucb_score = beta * sigma - mu
    
    # Apply linear cost penalty
    # Lower cost regions (x < 0) are favored.
    score = ucb_score / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
