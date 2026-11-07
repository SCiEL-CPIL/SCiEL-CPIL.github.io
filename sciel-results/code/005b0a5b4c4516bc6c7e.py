import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # LCB (Lower Confidence Bound) for minimization
    # We want to minimize mu - beta * sigma
    # So we maximize (beta * sigma - mu)
    # Divided by cost for cost-awareness
    
    # Beta controls exploration vs exploitation
    # Higher beta -> more exploration
    # Rosenbrock has a narrow valley, so we need to be careful not to wander too much
    # But EI can get stuck if the GP is slightly off. LCB is more robust to GP errors.
    
    beta = 3.0
    
    # Ensure sigma is not zero to avoid division issues, though we multiply here
    # mu and sigma are arrays of shape (2000,)
    
    # Calculate LCB score
    # lcb = mu - beta * sigma
    # score = -lcb / cost = (beta * sigma - mu) / cost
    
    # Add a small epsilon to cost to avoid division by zero (cost is always > 1)
    score = (beta * sigma - mu) / (cost + 1e-9)
    
    # Return finite scores
    return score
