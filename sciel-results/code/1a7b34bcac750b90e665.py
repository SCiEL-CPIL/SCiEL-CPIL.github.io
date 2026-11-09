import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (POI) acquisition function.
    
    Parameters:
    mu: array of shape (2000,) - Mean of the GP
    sigma: array of shape (2000,) - Standard deviation of the GP
    f_best: float - Best observed value so far
    cost: array of shape (2000,) - Cost of evaluating each point
    
    Returns:
    scores: array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Small epsilon to avoid division by zero
    eps = 1e-9
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score: (f_best - mu) / sigma
    # We are minimizing, so we want f(x) < f_best
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Clip to [0, 1] to handle numerical precision issues
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost-aware score: POI / cost
    # Higher POI is better, lower cost is better
    scores = poi / (cost + eps)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
