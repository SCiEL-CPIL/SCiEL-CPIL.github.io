import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB with exponential cost suppression.
    
    Score = (kappa * sigma - mu) * exp(-k * cost)
    
    Args:
        mu: array of GP means, shape (N,)
        sigma: array of GP standard deviations, shape (N,)
        f_best: scalar, best function value observed so far
        cost: array of costs, shape (N,)
        
    Returns:
        scores: array of acquisition values, shape (N,), higher is better
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margin check
    if time.time() - start_time > 290:
        return np.zeros_like(mu)

    # Prevent division by zero (not strictly needed for UCB but good practice)
    # sigma is already non-negative
    
    # UCB parameter: balance between exploitation (mu) and exploration (sigma)
    # Higher kappa means more exploration.
    kappa = 2.0
    
    # Cost gate parameter
    # k=1.0 as per lesson hint (sharper trade-off vs lower k, or balanced vs higher k)
    k = 1.0
    
    # Calculate UCB score for minimization:
    # We want to minimize f(x). The optimistic estimate is mu - kappa * sigma.
    # To maximize the acquisition, we use the negative of this estimate.
    # Score_base = -(mu - kappa * sigma) = kappa * sigma - mu
    score_base = kappa * sigma - mu
    
    # Apply exponential cost gate
    cost_gate = np.exp(-k * cost)
    
    scores = score_base * cost_gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final time check
    if time.time() - start_time > 295:
         pass
         
    return scores
