import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle sigma close to 0 to avoid division by zero
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) is the CDF, phi(z) is the PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0)
    
    # Cost-aware: Divide by cost
    # Ensure cost is not zero
    cost_safe = np.where(cost > 1e-8, cost, 1e-8)
    
    scores = ei / cost_safe
    
    # Handle any numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
