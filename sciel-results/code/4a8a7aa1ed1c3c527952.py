"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blend of Expected Improvement (EI) and Probability of Improvement (PI).
    This balances exploitation (finding better points) with exploration (uncertainty).
    
    The blend is weighted towards PI to break stagnation in local minima,
    while retaining some EI signal for steady progress.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (n,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Blend: 60% PI + 40% EI
    # PI focuses on direct probability of improvement, breaking local minima
    # EI provides a magnitude signal for how much improvement is expected
    scores = 0.6 * pi + 0.4 * ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
