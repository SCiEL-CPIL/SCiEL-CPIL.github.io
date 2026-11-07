import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Local Intensive Expected Improvement (LIEI)
    # Clip the improvement term to be non-negative to suppress exploration
    # in regions where the GP predicts values worse than the current best.
    # This forces the search to focus on high-probability improvement zones.
    
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Standard EI formula: E[max(f_best - Y, 0)] where Y ~ N(mu, sigma^2)
    # Formula: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # LIEI: Clip (f_best - mu) to be non-negative
    # This makes the linear term zero where mu > f_best
    mean_diff_clipped = np.maximum(f_best - mu, 0.0)
    
    ei = mean_diff_clipped * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    return ei
