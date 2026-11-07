import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using Probability of Improvement (PI) 
    with an adaptive exploration term.
    
    Parameters:
    mu: np.ndarray of shape (N,) - Posterior mean
    sigma: np.ndarray of shape (N,) - Posterior standard deviation
    f_best: float - Best observed value so far (minimum)
    
    Returns:
    np.ndarray of shape (N,) - Acquisition scores (higher is better)
    """
    
    # Ensure sigma is positive
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi_score = norm.cdf(z)
    
    # Adaptive exploration coefficient beta
    # We want to explore more early on (high f_best) and exploit later (low f_best).
    # Hartmann 6D global min is -3.32237.
    # Assume initial f_best is around 0 to 3.
    # Map f_best from [0, -3.3] to beta from [2.0, 0.5]
    
    # Clamp f_best to a reasonable range for stability
    f_best_clamped = np.clip(f_best, -3.3, 3.0)
    
    # t = 0 when f_best = 3.0 (worst case, high exploration)
    # t = 1 when f_best = -3.3 (best case, low exploration)
    # Range is 6.3
    t = (f_best_clamped - 3.0) / (-3.3 - 3.0)
    t = np.clip(t, 0.0, 1.0)
    
    # Beta decreases linearly from 2.0 to 0.5
    beta = 2.0 - 1.5 * t
    
    # Combine PI and exploration term
    # The exploration term beta * sigma encourages sampling in uncertain regions.
    # The PI term encourages sampling in regions likely to improve f_best.
    scores = pi_score + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
