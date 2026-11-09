import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Parameters:
    mu: array-like, shape (n_candidates,)
        Mean of the Gaussian Process posterior.
    sigma: array-like, shape (n_candidates,)
        Standard deviation of the Gaussian Process posterior.
    f_best: float
        The best (minimum) value observed so far.
        
    Returns:
    scores: array-like, shape (n_candidates,)
        Expected Improvement scores. Higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - f(x)
    z = (f_best - mu) / sigma
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # To avoid numerical issues with very large z (where Phi(z) ~ 1),
    # we can rely on scipy's stability, but let's be careful.
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei_scores = (f_best - mu) * cdf_vals + sigma * pdf_vals
    
    # Ensure finite scores
    ei_scores = np.nan_to_num(ei_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei_scores
