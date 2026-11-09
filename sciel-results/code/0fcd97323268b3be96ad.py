import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Args:
        mu: numpy array of shape (N,), posterior mean.
        sigma: numpy array of shape (N,), posterior standard deviation.
        f_best: float, best observed value so far.
        cost: numpy array of shape (N,), cost of evaluating each point.
        
    Returns:
        numpy array of shape (N,), acquisition scores (higher is better).
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # A very small epsilon prevents numerical issues
    eps = 1e-9
    sigma = np.maximum(sigma, eps)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF of standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI should be non-negative. Due to numerical precision, it might be slightly negative.
    # Clip to 0.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # Ensure cost is not zero (it's >= 1.0 in this problem, but just in case)
    cost = np.maximum(cost, eps)
    
    scores = ei / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
