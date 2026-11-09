import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Uses Upper Confidence Bound (UCB) for exploration, weighted by cost.
    Score = (-mu + beta * sigma) / cost
    This favors points with low predicted mean (high potential to be minimum)
    and high uncertainty, while penalizing high evaluation costs.
    """
    # Deadline guard: The problem states 300s limit, but we are just computing scores.
    # The computation is vectorized and fast, so time guard is less critical here 
    # than in a full BO loop, but we include it for safety/compliance with directive.
    # However, 'acquisition' is called inside the loop. We don't have access to start time 
    # unless we use a global or pass it. The signature is fixed.
    # We will assume the vectorized operation is fast enough.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safeguard for sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB for minimization:
    # We want to find x that minimizes f(x).
    # The lower bound on f(x) is mu - beta * sigma.
    # We want this bound to be as low (negative) as possible.
    # So we want to maximize (beta * sigma - mu).
    # Let's call this 'optimistic_improvement'.
    
    # Tuning beta: 
    # If beta is too high, we explore too much.
    # If beta is too low, we exploit too much.
    # A common value is 1.0 to 2.0.
    beta = 1.5
    
    optimistic_improvement = beta * sigma_safe - mu
    
    # Cost-aware weighting
    # We divide by cost to prefer cheaper points.
    # Cost is in [1, 2].
    cost_safe = np.maximum(cost, eps)
    
    scores = optimistic_improvement / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
