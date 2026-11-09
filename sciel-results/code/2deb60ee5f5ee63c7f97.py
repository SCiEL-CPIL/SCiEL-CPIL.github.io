# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Aggressive Probability of Improvement (PI) acquisition function.
    
    Uses a significant improvement threshold (xi) to avoid settling for
    modest gains and encourage exploration of regions with high potential
    for large improvements.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the Gaussian process posterior at the candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the Gaussian process posterior at the candidate points.
    f_best : float
        The lowest observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher scores indicate
        points that are more likely to be evaluated next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Define a threshold for "significant" improvement.
    # The function minimum is ~-2.53. A step of 0.5 is significant.
    # This forces the optimizer to ignore points that are only slightly better
    # than f_best, pushing it to explore for deeper minima.
    xi = 0.5
    
    # Compute z-score for Significant PI:
    # We want P(f(x) < f_best - xi)
    target = f_best - xi
    z = (target - mu) / sigma
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Significant Improvement
    scores = norm.cdf(z_clipped)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
