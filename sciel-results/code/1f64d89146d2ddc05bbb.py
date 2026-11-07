import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI) normalized by cost.
    
    Args:
        mu: Mean of the GP surrogate (shape: (2000,))
        sigma: Standard deviation of the GP surrogate (shape: (2000,))
        f_best: The best observed function value so far (scalar)
        cost: The cost of evaluating each candidate (shape: (2000,))
        
    Returns:
        scores: Acquisition scores for each candidate (shape: (2000,))
    """
    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for Probability of Improvement
    # We want P(f(x) < f_best)
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: Higher is better
    # We divide by cost to prefer cheaper evaluations
    scores = poi / cost_safe
    
    # Return finite scores
    return scores
