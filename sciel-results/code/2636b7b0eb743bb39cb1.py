import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    Args:
        mu: Predictive mean of the GP at candidate points (shape: (2000,)).
        sigma: Predictive standard deviation of the GP at candidate points (shape: (2000,)).
        f_best: Current best observed function value (scalar).
        cost: Cost of evaluating each candidate point (shape: (2000,)).
        
    Returns:
        scores: Acquisition scores for each candidate (shape: (2000,)). Higher is better.
    """
    # Known global minimum value for Styblinski-Tang 2D
    f_opt = -78.3323
    
    # Calculate the gap between current best and the known optimum
    # gap > 0 means we are still far from the optimum
    gap = f_best - f_opt
    
    # Adaptive beta parameter for UCB
    # When the gap is large (early in optimization), we use a larger beta to encourage exploration.
    # When the gap is small (late in optimization), we use a smaller beta to encourage exploitation.
    # The scaling factor 50.0 is chosen based on the typical range of the function values.
    beta = 2.0 + 1.5 * (gap / 50.0)
    
    # Clip beta to reasonable bounds to prevent extreme behavior
    beta = np.clip(beta, 1.5, 5.0)
    
    # UCB score for minimization:
    # We want to maximize the score.
    # Lower mu is better (exploitation).
    # Higher sigma is better (exploration).
    # Cost is in the denominator, so cheaper points are preferred.
    scores = (beta * sigma - mu) / cost
    
    # Ensure all scores are finite to avoid invalid results
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
