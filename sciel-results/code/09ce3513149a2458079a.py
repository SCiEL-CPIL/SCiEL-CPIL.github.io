import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    
    Uses a Predictive Entropy Search (PES)-inspired score that rewards
    points with high uncertainty that have a non-negligible probability
    of containing a new best value. The score is:
    
        score = sigma * (1 - Phi((f_best - mu) / sigma))
    
    This balances exploration (high sigma) and exploitation (low mu relative
    to f_best) in a principled way.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Z score
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues
    Z_clipped = np.clip(Z, -10, 10)
    
    # Compute survival function: P(f(x) < f_best) = 1 - Phi(Z)
    survival = norm.sf(Z_clipped)
    
    # PES-inspired score: sigma * P(f(x) < f_best)
    # This rewards points with high uncertainty AND high probability of improvement
    scores = sigma * survival
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
