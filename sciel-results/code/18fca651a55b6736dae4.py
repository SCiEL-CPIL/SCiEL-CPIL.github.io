import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Args:
        mu: Mean of the GP surrogate, shape (2000,)
        sigma: Standard deviation of the GP surrogate, shape (2000,)
        f_best: Best observed function value (scalar)
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,), higher is better.
    """
    # UCB parameter: balances exploration (sigma) and exploitation (-mu)
    # A value of 2.5 to 4.0 is common for low-budget BO.
    kappa = 3.0
    
    # UCB score for minimization: -mu + kappa * sigma
    # We want to maximize this value.
    ucb = -mu + kappa * sigma
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: UCB per unit cost
    scores = ucb / cost_safe
    
    # Handle numerical issues (NaN, Inf)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
