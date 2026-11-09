import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate Expected Improvement
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Add small epsilon to sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    # norm.cdf is stable, norm.pdf can underflow for large |z|
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Cost-aware scoring
    # We want to penalize high cost, but maybe not too strongly.
    # Original: EI / cost
    # Modified: EI / (1 + 0.5 * (cost - 1))
    # This reduces the range of the penalty from [1, 2] to [1, 1.5]
    
    cost_penalty = 1.0 + 0.5 * (cost - 1.0)
    
    scores = ei / cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
