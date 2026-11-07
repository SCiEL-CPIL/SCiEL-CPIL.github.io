import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using GP-UCB (Upper Confidence Bound)
    for minimization.
    
    The score is defined as:
    Score(x) = (-mu(x) + beta * sigma(x)) / Cost(x)
    
    We maximize this score. A high score indicates a point with a low expected
    function value (good for minimization) and/or high uncertainty, normalized
    by the evaluation cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Beta parameter for UCB. 
    # A value of 1.0 is a standard starting point for balanced exploration/exploitation.
    beta = 1.0
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization is typically mu - beta * sigma.
    # Since we want to MAXIMIZE the acquisition score to select the next point,
    # and we want to MINIMIZE the function, we invert the UCB logic.
    # We want to find points where (mu - beta * sigma) is LOW.
    # So we maximize -(mu - beta * sigma) = -mu + beta * sigma.
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize by cost to prefer cheaper evaluations
    # Cost is guaranteed to be >= 1.0 based on the problem description (1 + mean(scaled_x))
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ucb_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Replace -inf with a very small number to ensure finite output as required
    # The task says "Return 2000 finite scores".
    # Using -1e10 is a safe large negative number.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
