import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Calculate z-score for probability of improvement
    # z = (f_best - mu) / sigma
    # For minimization, we want mu < f_best, so f_best - mu > 0, z > 0.
    # PI = P(f(x) < f_best) = Phi(z)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi_scores = norm.cdf(z)
    
    # Exploration bonus
    # Normalize sigma to [0, 1] based on the max sigma in the batch
    max_sigma = np.max(sigma)
    if max_sigma > eps:
        sigma_norm = sigma / max_sigma
    else:
        sigma_norm = np.zeros_like(sigma)
        
    # Weighted combination
    # Primary signal: Probability of Improvement
    # Secondary signal: Uncertainty (Exploration)
    lambda_explore = 0.1
    scores = pi_scores + lambda_explore * sigma_norm
    
    return scores
