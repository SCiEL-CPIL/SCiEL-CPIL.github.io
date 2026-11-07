"""New acquisition function using Cost-Adjusted Upper Confidence Bound."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) normalized by cost.
    
    Score(x) = (beta * sigma(x) - mu(x)) / cost(x)
    
    For minimization, we maximize the potential for finding a lower value.
    -mu(x) rewards low predicted values (exploitation).
    beta * sigma(x) rewards high uncertainty (exploration).
    Dividing by cost penalizes expensive evaluations.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # UCB parameter for exploration/exploitation trade-off
    # A value of 2.5 is a common heuristic for small-budget BO
    beta = 2.5
    
    # Calculate UCB score for minimization:
    # We want to maximize (beta * sigma - mu)
    # This is equivalent to minimizing (mu - beta * sigma)
    ucb_score = beta * sigma - mu
    
    # Normalize by cost
    scores = ucb_score / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
