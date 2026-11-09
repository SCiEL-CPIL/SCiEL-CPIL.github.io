import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) normalized by a power of the evaluation cost.
    
    Parameters:
    mu (np.ndarray): Mean predictions from the GP, shape (N,)
    sigma (np.ndarray): Standard deviation predictions from the GP, shape (N,)
    f_best (float): The best (lowest) observed objective value so far.
    cost (np.ndarray): Cost of evaluating each point, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, higher is better, shape (N,)
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    
    # Numerical stability
    eps = 1e-9
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, eps)
    
    # Check for early exit if we're running out of time
    if time.time() - t_start > 280:
        return np.zeros_like(mu)
    
    # Exploration parameter xi for EI.
    # Small xi to focus on exploitation while allowing some exploration.
    xi = 0.01
    
    # Calculate z-score for modified EI
    z = (f_best - mu - xi * sigma_safe) / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement with xi
    ei = (f_best - mu - xi * sigma_safe) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # Divide by cost^gamma. gamma < 1 makes the score less sensitive to cost.
    gamma = 0.5
    cost_safe = np.maximum(cost, eps)
    scores = ei / np.power(cost_safe, gamma)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() - t_start > 290:
        return scores
    
    return scores
