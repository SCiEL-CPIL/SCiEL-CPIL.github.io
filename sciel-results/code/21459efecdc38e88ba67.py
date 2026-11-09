"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function.
    
    Args:
        mu: Array of shape (N,) containing the GP mean predictions.
        sigma: Array of shape (N,) containing the GP standard deviation predictions.
        f_best: Scalar, the best observed value so far (minimum).
        
    Returns:
        Array of shape (N,) containing the PI scores. Higher scores indicate
        a higher probability of finding a point better than f_best.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    # We use a small epsilon to handle points where the GP is very certain
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Calculate the z-score for the probability of improvement
    # We want P(f(x) < f_best)
    # If f(x) ~ N(mu, sigma^2), then P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to a reasonable range to ensure numerical stability of norm.cdf
    # norm.cdf is stable for large values, but clipping prevents extreme
    # floating point issues in edge cases
    z_clipped = np.clip(z, -10, 10)
    
    # Compute the Probability of Improvement
    pi_scores = norm.cdf(z_clipped)
    
    # Ensure the output is finite and within [0, 1]
    pi_scores = np.clip(pi_scores, 0.0, 1.0)
    pi_scores = np.where(np.isfinite(pi_scores), pi_scores, 0.0)
    
    return pi_scores
# EVOLVE-BLOCK-END
