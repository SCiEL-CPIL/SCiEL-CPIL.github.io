import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: scalar, the lowest observed value so far
    - cost: array of shape (2000,) with evaluation costs for each candidate
    
    Returns:
    - scores: array of shape (2000,) with acquisition values (higher = evaluate next)
    """
    # Ensure sigma and cost are safe for division/multiplication
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB parameter tuned from knowledge (1.0 -> 0.8 improved score)
    beta = 0.8
    
    # UCB score for minimization:
    # We want to minimize f(x).
    # GP predicts f(x) ~ N(mu, sigma^2).
    # We prefer low mu (exploitation) and high sigma (exploration).
    # Score = -mu + beta * sigma
    ucb_scores = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost
    # Higher score per unit cost is better
    scores = ucb_scores / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
