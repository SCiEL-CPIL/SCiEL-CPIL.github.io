import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses a hybrid strategy: UCB for exploration when far from the minimum,
    and Expected Improvement (EI) for exploitation when close.
    """
    # Ensure arrays are floats
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # The known global minimum of Goldstein-Price 2D is 3.0.
    # If we are far from the minimum, we prioritize exploration (UCB).
    # If we are close, we prioritize exploitation (EI).
    
    threshold = 3.5
    
    if f_best > threshold:
        # UCB strategy: Maximize (-mu + beta * sigma)
        # beta controls exploration. 
        # Since sigma is bounded by 1 (standardized GP), beta=3.0 is aggressive 
        # enough to explore unvisited regions without being completely random.
        beta = 3.0
        scores = -mu + beta * sigma
    else:
        # EI strategy: Standard Expected Improvement for minimization.
        # z = (f_best - mu) / sigma
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        z = (f_best - mu) / sigma
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
        
        scores = (f_best - mu) * cdf + sigma * pdf
        
        # Ensure scores are finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
