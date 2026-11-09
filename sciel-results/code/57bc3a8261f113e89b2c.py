import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) with cost awareness
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle sigma close to 0 to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scoring: higher EI per cost is better
    # cost is given as 1 + mean(x_scaled), so it's at least 1.
    # We divide by cost to penalize expensive points.
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
