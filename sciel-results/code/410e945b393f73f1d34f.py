import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) for minimization.
    
    Args:
        mu: array of mean predictions (shape 2000,)
        sigma: array of std deviations (shape 2000,)
        f_best: scalar, best observed value so far
        cost: array of evaluation costs (shape 2000,)
        
    Returns:
        array of scores (higher is better)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We maximize EI per unit cost
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost normalization
    scores = ei / cost
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=0.0)
    
    return scores
