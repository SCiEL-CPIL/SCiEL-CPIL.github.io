"""Rewritten program for Bayesian optimization of Hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses a Thompson Sampling proxy based on the normalized improvement (z-score)
    to balance exploration and exploitation. This encourages sampling points
    with high potential relative to their uncertainty.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and add small epsilon to avoid division by zero
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Thompson Sampling proxy: Normalized Improvement (Z-score)
    # High score when mu is significantly lower than f_best relative to sigma
    z_scores = (f_best - mu) / sigma_safe
    
    # The z-score can be very large. To keep the scale manageable and stable,
    # we can use the CDF (Probability of Improvement) which is bounded [0, 1]
    # or clip the z-scores. 
    # The directive suggests the score itself. However, raw z-scores can lead to 
    # numerical issues or overwhelming the selection if one point is an outlier.
    # Let's use the CDF of the z-score, which is the Probability of Improvement (PI).
    # PI is a robust structural change from EI and aligns with the "probability" 
    # aspect of Thompson Sampling (sampling from posterior).
    
    scores = norm.cdf(z_scores)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 325:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
