import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = None


def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Sharpened Expected Improvement with adaptive exploration per unit cost.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: scalar - best observed function value
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the 300s limit, return a simple fallback
    if _get_elapsed_time() > 290:
        mu = np.asarray(mu)
        cost = np.asarray(cost)
        cost_safe = np.maximum(cost, 1e-9)
        scores = -mu / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standard normal CDF/PDF arguments
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement components
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    # Exploit term: (f_best - mu) * cdf(z)
    exploit = (f_best - mu) * cdf_val
    
    # Explore term: sigma * pdf(z)
    explore = sigma_safe * pdf_val
    
    # Sharpened EI: reduce the exploration weight to be more greedy
    # alpha controls exploration vs exploitation balance
    # Use a slightly lower alpha than 0.5 to be more exploitative
    alpha = 0.4
    ei_sharp = exploit + alpha * explore
    
    # Cost-aware: score per unit cost
    score = ei_sharp / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
