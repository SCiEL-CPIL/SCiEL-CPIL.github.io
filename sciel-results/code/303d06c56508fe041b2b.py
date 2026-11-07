import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB)
    with a linear cost penalty.
    
    Strategy:
    1. Compute LCB: mu - beta * sigma.
       This balances exploitation (low mu) and exploration (high sigma).
       Unlike PI, LCB is continuous and sensitive to the magnitude of improvement.
    2. Apply a linear cost penalty: - lambda * cost.
       This structurally differs from multiplicative penalties by allowing
       high-potential points to offset high costs if the mean improvement is sufficient.
    3. Combine additively.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # LCB Parameters
    # beta controls exploration-exploitation trade-off
    # A moderate beta (1.0-2.0) is typical for UCB/LCB in BO
    # We use 1.5 to encourage some exploration in rugged landscapes
    beta = 1.5
    
    # Cost Penalty Parameter
    # Cost is in [1, 2].
    # Objective values (Griewank) are typically in [0, 1].
    # We want the cost to be a significant but not dominant factor.
    # If lambda is too high, we only pick the cheapest points (corners).
    # If lambda is too low, we ignore cost.
    # Let's try lambda = 0.1. 
    # A cost difference of 1.0 (min to max) penalizes by 0.1.
    # This is comparable to a moderate improvement in the objective.
    lambda_cost = 0.1
    
    # Compute LCB
    # We are minimizing, so we want low mu.
    # LCB = mu - beta * sigma
    lcb = mu - beta * sigma_safe
    
    # Apply cost penalty
    # Lower cost is better, so we subtract cost * lambda
    score = lcb - lambda_cost * cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    # If all scores are identical (e.g., very early on or flat GP), add small noise
    if np.max(score) - np.min(score) < 1e-15:
        score = score + np.random.randn(len(score)) * 1e-10
        
    return score
