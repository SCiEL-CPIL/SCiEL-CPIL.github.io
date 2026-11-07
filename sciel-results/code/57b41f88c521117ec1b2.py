import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # EI is a standard acquisition function that balances exploration and exploitation
    # by maximizing the expected reduction in the best observed value.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score: how many standard deviations f_best is above mu
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement formula for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_scores = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Handle numerical issues: ensure finite scores
    ei_scores = np.nan_to_num(ei_scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return ei_scores
