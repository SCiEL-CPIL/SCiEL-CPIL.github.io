import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Use Upper Confidence Bound for minimization
    # We want to minimize f(x).
    # The "value" of a point is how much better it could be than f_best.
    # UCB lower bound for minimization: mu - beta * sigma
    # Improvement potential: f_best - (mu - beta * sigma) = f_best - mu + beta * sigma
    
    # Clip sigma to avoid division by zero or extreme values
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Beta parameter for exploration/exploitation trade-off
    # A higher beta encourages exploration.
    # In early BO, we want to explore.
    beta = 2.0
    
    # Calculate UCB-based improvement
    # Note: If mu is already better than f_best, f_best - mu is negative.
    # But we add beta * sigma, so it might become positive.
    # We should ensure we only consider improvements.
    # If f_best - mu + beta * sigma is negative, the expected improvement is 0 (or negative, which we cap at 0).
    improvement = f_best - mu + beta * sigma_safe
    
    # Cap at 0 to avoid selecting points that are definitely worse
    # (though with beta * sigma, it's rare for it to be negative if sigma is large enough)
    improvement = np.maximum(improvement, 0)
    
    # Cost-aware: divide by cost
    # Ensure cost is not zero
    cost_safe = np.clip(cost, 1e-8, None)
    
    scores = improvement / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
