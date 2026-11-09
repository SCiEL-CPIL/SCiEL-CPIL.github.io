import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    array of shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Use Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    # For a normal distribution f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    score = ei / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
