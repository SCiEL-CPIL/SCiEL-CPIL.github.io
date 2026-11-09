import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Returns 2000 finite scores (higher = evaluate next).
    Uses Cost-Adjusted Expected Improvement (EI / Cost).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-12)
    
    # Initialize scores
    scores = np.zeros_like(mu)
    
    # Handle sigma = 0 or very small
    # If sigma is small, we can approximate EI
    eps = 1e-12
    small_sigma = sigma < eps
    
    if np.any(small_sigma):
        # For small sigma, EI is approx f_best - mu if mu < f_best, else 0
        improvement = f_best - mu[small_sigma]
        scores[small_sigma] = np.maximum(improvement, 0.0) / cost_safe[small_sigma]
        
    # For normal sigma
    if np.any(~small_sigma):
        mu_valid = mu[~small_sigma]
        sigma_valid = sigma[~small_sigma]
        cost_valid = cost_safe[~small_sigma]
        
        z = (f_best - mu_valid) / sigma_valid
        # Calculate EI
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        ei = (f_best - mu_valid) * norm.cdf(z) + sigma_valid * norm.pdf(z)
        
        # Ensure EI is non-negative (numerical errors can make it slightly negative)
        ei = np.maximum(ei, 0.0)
        
        scores[~small_sigma] = ei / cost_valid
        
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
