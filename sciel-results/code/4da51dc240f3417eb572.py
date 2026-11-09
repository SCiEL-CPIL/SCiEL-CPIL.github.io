import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Probability of Improvement (POI): score = P(f(x) < f_best).
    This probabilistic approach balances exploration and exploitation by
    favoring regions where the posterior probability of beating the current
    best is high.
    """
    start_time = time.time()
    timeout = 320.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    n = len(mu)
    
    # Ensure sigma is positive for computations
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-scores for Probability of Improvement
    # We want f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Score is the CDF of the standard normal distribution at z
    scores = norm.cdf(z)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - start_time > timeout:
        return scores
    
    return scores
