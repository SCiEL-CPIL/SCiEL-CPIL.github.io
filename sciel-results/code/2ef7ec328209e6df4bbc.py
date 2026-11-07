import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Calculate Expected Improvement
        z = (f_best - mu) / (sigma + 1e-9)
        # Avoid log(0)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 1e-10)
        
        # Log-EI to compress dynamic range
        log_ei = np.log(ei)
        
        # Cost-aware score
        scores = log_ei / cost
        
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        return scores
