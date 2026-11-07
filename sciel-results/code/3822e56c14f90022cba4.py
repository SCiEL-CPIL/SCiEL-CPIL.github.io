import numpy as np
import time

# Deadline guard
TIME_BUDGET = 320.0
START_TIME = None

def _elapsed():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Returns scores to be maximized.
    score = beta * sigma - mu
    
    A higher score indicates a candidate that is either very low (exploit) 
    or has high uncertainty (explore).
    
    Parameters:
    mu: array of shape (N,), posterior mean in objective units.
    sigma: array of shape (N,), posterior standard deviation in objective units.
    f_best: float, current best observed value.
    
    Returns:
    array of shape (N,), finite scores.
    """
    # Deadline guard
    if _elapsed() > TIME_BUDGET:
        # Fallback: Simple UCB with small beta
        beta = 0.1
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = beta * sigma_safe - mu
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # UCB parameter
    # Using a higher exploration constant (k=3.0) to aggressively target high-variance regions.
    # This helps in navigating the high-gradient ridges surrounding the isolated minima of Michalewicz 2D.
    beta = 3.0
    
    # Calculate UCB score for maximization
    # We want to find low values of f.
    # Lower bound of f is approx mu - beta * sigma.
    # We maximize the potential to be low, so we maximize - (mu - beta * sigma)
    # which is beta * sigma - mu.
    scores = beta * sigma_safe - mu_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
