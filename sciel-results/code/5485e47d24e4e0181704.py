import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with an exploration parameter xi,
    normalized by a power of the evaluation cost.
    
    Parameters:
    mu (np.ndarray): Mean predictions from the GP, shape (N,)
    sigma (np.ndarray): Standard deviation predictions from the GP, shape (N,)
    f_best (float): The best (lowest) observed objective value so far.
    cost (np.ndarray): Cost of evaluating each point, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, higher is better, shape (N,)
    """
    start_time = time.time()
    
    # Numerical stability
    eps = 1e-9
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, eps)
    
    # Exploration parameter xi. 
    # xi > 0 encourages exploring regions with high uncertainty even if 
    # the mean is not significantly better than f_best.
    # Ackley is multimodal, so some exploration is beneficial.
    xi = 0.15
    
    # Calculate z-score for modified EI
    # z = (f_best - mu - xi * sigma) / sigma
    z = (f_best - mu - xi * sigma_safe) / sigma_safe
    
    # Calculate CDF and PDF of standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement with xi
    # EI = (f_best - mu - xi*sigma) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi * sigma_safe) * cdf + sigma_safe * pdf
    
    # EI should be non-negative, but numerical issues can cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # We divide by cost^gamma. 
    # gamma < 1 makes the score less sensitive to cost, allowing the GP 
    # uncertainty (EI) to dominate the selection more, which helps in 
    # navigating the multimodal landscape of Ackley.
    # cost is in [1, 2].
    gamma = 0.4
    
    cost_safe = np.maximum(cost, eps)
    scores = ei / np.power(cost_safe, gamma)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard: if we're taking too long, return a simple fallback
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        # Fallback: just use EI without cost adjustment
        fallback_scores = np.where(np.isfinite(ei), ei, 0.0)
        return fallback_scores
    
    return scores
