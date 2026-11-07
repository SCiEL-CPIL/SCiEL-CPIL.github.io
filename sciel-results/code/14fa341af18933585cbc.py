import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function for minimization.
    
    Args:
        mu: Array of shape (2000,) with GP mean predictions.
        sigma: Array of shape (2000,) with GP standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (2000,) with evaluation costs.
        
    Returns:
        Array of shape (2000,) with scores (higher is better).
    """
    # Ensure sigma is strictly positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: EI per unit cost
    # Cost is in [1, 2], so this is a reasonable normalization
    score = ei / cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
