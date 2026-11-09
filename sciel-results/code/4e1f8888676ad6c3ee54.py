import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) scores for candidate points.
    
    Parameters:
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (N,)
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (N,)
    f_best : float
        Current best (lowest) observed value.
        
    Returns:
    np.ndarray
        EI scores for each candidate, shape (N,)
    """
    # Ensure sigma is positive to avoid division by zero or invalid values
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the standard normal CDF and phi is the standard normal PDF
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure scores are finite
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
