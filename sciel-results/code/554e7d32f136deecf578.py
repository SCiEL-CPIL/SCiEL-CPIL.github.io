import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) and phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We want to maximize EI
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Probability of Improvement
    poi = cdf
    
    # Blend: primarily EI, but add a small PoI term to break ties and handle flat regions
    # This helps when EI is near zero but there's a chance of improvement
    scores = ei + 0.1 * poi * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
