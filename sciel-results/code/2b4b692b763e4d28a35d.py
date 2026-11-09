import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu: array of mean predictions (shape N,)
    sigma: array of standard deviations (shape N,)
    f_best: float, the best (lowest) function value observed so far
    
    Returns:
    scores: array of EI values (shape N,), higher is better.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # and numerical issues in norm.pdf/cdf.
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF of standard normal, phi is PDF of standard normal.
    
    # Using scipy.stats.norm for accuracy
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
