import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use Expected Improvement with xi=0
    # This balances exploration and exploitation based on the posterior distribution.
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    scores = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
