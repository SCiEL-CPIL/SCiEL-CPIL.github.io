import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use Expected Improvement (EI) for minimization.
    # EI(x) = E[max(f_best - f(x), 0)]
    # For a Gaussian prediction N(mu, sigma^2), 
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma.
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # norm.cdf(z) computes Phi(z)
    # norm.pdf(z) computes phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-awareness: Maximize the expected improvement per unit cost.
    # This penalizes expensive evaluations proportionally to their cost.
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
