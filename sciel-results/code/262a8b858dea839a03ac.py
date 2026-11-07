import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviation predictions.
        f_best: Scalar, the best objective value observed so far.
        cost: (2000,) array of evaluation costs (1 + mean(x)).
        
    Returns:
        (2000,) array of acquisition scores.
    """
    # Convert to numpy arrays for consistency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Calculate z-score for EI
    z = (f_best - mu) / safe_sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Clip negative EI to zero (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # Use superlinear cost penalty to strongly favor cheaper points
    alpha = 2.0
    score = ei / np.power(safe_cost, alpha)
    
    # Ensure output is finite and real
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
