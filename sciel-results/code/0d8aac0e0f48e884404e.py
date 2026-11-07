import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP at each candidate point.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP at each candidate point.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    # Calculate the z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment
    # We divide by cost to prefer evaluations that yield high improvement per unit cost.
    # Add epsilon to avoid division by zero.
    cost_safe = np.maximum(cost, 1e-6)
    
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
