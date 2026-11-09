"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI).
    
    Parameters:
    mu : numpy array of shape (N,)
        Mean of the GP posterior.
    sigma : numpy array of shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        The best observed value so far (minimum).
        
    Returns:
    scores : numpy array of shape (N,)
        Probability of improvement for each candidate. Higher is better.
    """
    start_time = time.time()
    deadline = start_time + 290  # 290s safety margin within 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle potential NaNs or infinities in inputs
    # If sigma is 0 or negative, it's invalid. We replace with a tiny epsilon.
    # If mu is nan, the score should be 0 (or very low).
    
    # Calculate z-score: (f_best - mu) / sigma
    # To avoid division by zero, add a small epsilon to sigma
    eps = 1e-10
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # If sigma was originally <= eps, the uncertainty is negligible.
    # PI is 1 if mu < f_best, 0 if mu > f_best, 0.5 if mu == f_best.
    # The formula (f_best - mu) / sigma will handle this naturally if sigma is eps,
    # resulting in huge positive or negative z, which maps to 1 or 0.
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # Phi is the standard normal CDF
    scores = norm.cdf(z)
    
    # Ensure scores are finite
    # norm.cdf returns values in (0, 1), so they are always finite.
    # However, if mu or sigma contained NaN, z would be NaN, and cdf(NaN) is NaN.
    # We replace NaNs with 0.0 (no improvement expected)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Check time
    if time.time() > deadline:
        # If we are running out of time, we return the current scores.
        # Since PI is vectorized and fast, this is unlikely to be hit.
        pass
        
    return scores
# EVOLVE-BLOCK-END
