import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI)
    # z = (f_best - mu) / sigma
    # PI = P(f(x) < f_best) = Phi(z)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Cost-aware score: PI divided by cost
    # This prioritizes points with high probability of improvement and low cost
    scores = pi / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
