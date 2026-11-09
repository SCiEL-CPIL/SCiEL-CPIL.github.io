import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) normalized by cost.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed value so far (scalar)
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition values, shape (2000,), higher is better.
    """
    # Ensure sigma is not zero to avoid division by zero in Z calculation
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate the Z-score for Expected Improvement
    # We are minimizing, so improvement is f_best - f(x)
    diff = f_best - mu
    Z = diff / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Using scipy.stats.norm for CDF and PDF
    EI = diff * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # EI should theoretically be non-negative, but numerical errors might make it slightly negative
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: EI per unit cost
    # This prioritizes points that offer the most expected improvement per unit of cost
    scores = EI / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
