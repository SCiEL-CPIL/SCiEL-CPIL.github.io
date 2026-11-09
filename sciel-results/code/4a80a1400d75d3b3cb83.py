import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses a Probabilistic Lower Bound (PLB) style score: maximize the probability
    that the true function value is better than a dynamic threshold.
    The score is (threshold - mu) / sigma, which is the z-score for improvement.
    The threshold is set slightly below f_best to encourage finding new minima.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0
    
    # Dynamic threshold: slightly below f_best to encourage improvement
    # Use a small offset to avoid degenerate behavior when mu is very close to f_best
    # The offset scales with the typical scale of sigma to be robust
    sigma_mean = np.mean(sigma)
    threshold_offset = 0.1 * sigma_mean
    threshold = f_best - threshold_offset
    
    # PLB score: (threshold - mu) / sigma
    # This maximizes P(f(x) < threshold), i.e., the probability of finding a point
    # that is better than the threshold (and thus likely better than f_best)
    scores = (threshold - mu) / sigma
    
    # Clip to prevent numerical issues with very small sigma
    scores = np.clip(scores, -1e6, 1e6)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
