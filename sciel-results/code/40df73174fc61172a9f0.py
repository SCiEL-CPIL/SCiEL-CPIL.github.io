import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB)
    # We want to minimize (mu - beta * sigma) subject to cost.
    # Score = -(mu - beta * sigma) / cost
    # Higher score means more desirable.
    
    beta = 2.0
    
    # Ensure cost is at least 1 to avoid division by zero or near-zero
    # The problem states cost = 1 + mean(x_scaled), so min cost is 1.
    # But just in case.
    cost_safe = np.maximum(cost, 1.0)
    
    # Calculate UCB value (lower is better)
    ucb_value = mu - beta * sigma
    
    # Score: higher is better
    # We want to select points with low ucb_value.
    # So we negate it.
    # Dividing by cost penalizes expensive points.
    score = -ucb_value / cost_safe
    
    return score
