import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) acquisition function for minimization.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Z-score: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    Z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    POI = norm.cdf(Z)
    
    # Ensure finite values
    POI = np.where(np.isfinite(POI), POI, 0.0)
    
    return POI
