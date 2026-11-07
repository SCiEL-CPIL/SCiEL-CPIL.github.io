import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is non-negative and not too small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for Expected Improvement
    # We want to minimize, so improvement is f_best - f(x)
    # E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # norm.cdf is Phi, norm.pdf is phi
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle cases where f_best - mu is negative (unlikely if f_best is min, but mu can be lower? No, f_best is min observed. mu is posterior mean. If mu < f_best, z is negative. EI should be 0 if we assume we know the min? 
    # Actually, if mu < f_best, the GP thinks there might be a lower value. 
    # Standard EI formula handles this. If z is very negative, cdf(z) ~ 0, pdf(z) ~ 0. EI ~ 0.
    
    # Ensure finite values
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
