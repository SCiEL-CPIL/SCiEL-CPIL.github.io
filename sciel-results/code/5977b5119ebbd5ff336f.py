"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: we must return within the 300s limit per evaluation.
_DEADLINE = 300.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates using POI
    weighted by a dynamic exploration factor.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP posterior
    sigma : ndarray, shape (2000,)
        Std of GP posterior
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Check time budget
    if _time_left() < 1.0:
        # Return a simple fallback if we're running out of time
        return np.zeros_like(mu)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Clip z to avoid numerical issues with very large values
    z = np.clip(z, -50, 50)
    
    # POI = Phi(z) = P(f(x) < f_best)
    POI = norm.cdf(z)
    
    # Dynamic exploration factor:
    # Use the global maximum uncertainty to normalize exploration.
    # Early on (high max sigma), we want more exploration; later, less.
    max_sigma = np.max(sigma)
    if max_sigma < 1e-12:
        max_sigma = 1e-12
    
    # Exploration weight: higher when uncertainty is high relative to global max
    # This creates a smooth transition from exploration to exploitation
    # Using a soft-max-like normalization
    exploration_weight = sigma / max_sigma  # in [0, 1]
    
    # Blend POI with exploration: 
    # score_base = POI * (1 - alpha) + exploration_weight * alpha
    # Where alpha adapts based on how much uncertainty remains
    # When max_sigma is high (early), alpha is higher (more exploration)
    # When max_sigma is low (late), alpha is lower (more exploitation)
    
    # Use a simple heuristic: alpha = min(0.5, max_sigma / (f_best + 1e-9))
    # But f_best can be close to 0, so use a safer formulation
    # alpha = min(0.5, max_sigma / (abs(f_best) + max_sigma + 1e-9))
    # This gives alpha in (0, 0.5), higher when uncertainty is large relative to best value
    
    alpha = min(0.5, max_sigma / (abs(f_best) + max_sigma + 1e-9))
    
    # Base score: blend of exploitation (POI) and exploration (uncertainty)
    base_score = POI * (1 - alpha) + exploration_weight * alpha
    
    # Ensure base_score is non-negative and finite
    base_score = np.maximum(base_score, 0.0)
    base_score = np.where(np.isfinite(base_score), base_score, 0.0)
    
    # Cost-aware: divide by cost
    scores = base_score / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
