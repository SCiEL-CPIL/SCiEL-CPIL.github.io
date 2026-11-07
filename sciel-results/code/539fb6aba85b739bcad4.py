import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function based on Expected Improvement (EI) per unit cost.
    """
    # Ensure sigma is non-negative and avoid division by zero
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Calculate score: EI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
