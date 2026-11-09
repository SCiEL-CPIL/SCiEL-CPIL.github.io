import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) for minimization, cost-aware.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle sigma == 0 or very small to avoid numerical issues
    eps = 1e-8
    mask = sigma > eps
    
    ei = np.zeros_like(mu)
    
    if np.any(mask):
        z = (f_best - mu[mask]) / sigma[mask]
        # Vectorized calculation
        # Note: norm.cdf and norm.pdf handle arrays
        cdf_val = norm.cdf(z)
        pdf_val = norm.pdf(z)
        ei[mask] = (f_best - mu[mask]) * cdf_val + sigma[mask] * pdf_val
        
    if np.any(~mask):
        # If sigma is effectively 0, the improvement is deterministic
        ei[~mask] = np.maximum(0, f_best - mu[~mask])
        
    # Cost-aware scaling: prioritize cheaper evaluations
    score = ei / cost
    
    return score
