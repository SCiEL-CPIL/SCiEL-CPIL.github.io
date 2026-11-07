import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) normalized by cost.
    
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
    
    # Calculate Z-score
    # We want E[max(f_best - f(x), 0)]
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (due to numerical errors, it might be slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-aware score: Higher is better
    # We divide by cost to prefer cheaper evaluations
    scores = ei / cost_safe
    
    # Return finite scores
    return scores
