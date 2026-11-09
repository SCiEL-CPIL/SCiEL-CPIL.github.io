import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # Expected Improvement
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Small exploration term to avoid premature convergence
        # Lambda is small to preserve EI's exploitation strength
        lambda_exp = 0.1
        scores = ei + lambda_exp * sigma
        
        # Ensure finite scores
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        
        return scores
