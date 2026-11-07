import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses Adaptive Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta(f_best) * sigma
    """
    start_time = time.time()
    deadline = 295.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-6)

    # Adaptive beta based on f_best
    # f_best starts high (~25) and goes to 0.
    # We want high exploration early, low exploration late.
    
    # Use a smooth decay
    # beta = min_beta + (max_beta - min_beta) * exp(-f_best / scale)
    # If f_best=25, exp(-25/scale) is small -> beta ~ min_beta. WRONG.
    # We want beta HIGH when f_best is HIGH.
    
    # beta = min_beta + (max_beta - min_beta) * (f_best / max_f)
    # If f_best=25, beta = max_beta.
    # If f_best=0, beta = min_beta.
    
    max_f = 25.0
    max_beta = 3.0  # High exploration
    min_beta = 0.5  # Low exploration
    
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    beta = min_beta + (max_beta - min_beta) * ratio
    
    # Calculate score
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        fallback_score = -np.where(np.isfinite(mu), mu, f_best)
        fallback_score = np.where(np.isfinite(fallback_score), fallback_score, -1e10)
        return fallback_score
        
    return score
