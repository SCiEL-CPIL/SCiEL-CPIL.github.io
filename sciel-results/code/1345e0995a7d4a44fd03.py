import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Draws a single sample from the GP posterior predictive distribution
    and selects the candidate that minimizes normalized regret.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP std dev predictions  
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,) - higher means evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Draw a single standard normal sample
    z = np.random.randn(len(mu))
    
    # Thompson Sampling: draw from posterior predictive
    theta = mu + sigma * z
    
    # Normalized regret: (theta - f_best) / sqrt(cost)
    # We want to MINIMIZE this, so we negate for "higher is better" scoring
    sqrt_cost = np.sqrt(cost)
    normalized_regret = (theta - f_best) / sqrt_cost
    
    # Negate so higher score = better (lower regret)
    scores = -normalized_regret
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
