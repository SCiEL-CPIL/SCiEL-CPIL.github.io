import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    For minimization, we maximize -mu + beta * sigma, penalized by cost.
    """
    # Exploration parameter
    beta = 1.5
    
    # Base UCB score for minimization (maximize this)
    # We want low mu, high sigma
    ucb_score = -mu + beta * sigma
    
    # Cost-aware adjustment
    # Normalize cost to be around 1.0? 
    # cost is 1 + mean(x). Mean(x) in [0,1]. So cost in [1, 2].
    # Dividing by cost penalizes expensive points.
    
    # Use sqrt(cost) or cost?
    # Let's try cost (linear) for stronger cost awareness.
    # If linear is too strong, sqrt is safer.
    # Let's stick to sqrt(cost) as it worked in the EI version, 
    # but the core is UCB now.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = ucb_score / np.sqrt(cost)
        
    # Replace NaN and Inf
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores
