"""Cost-aware acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Uses a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    normalized by cost, with careful handling of edge cases.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = more promising)
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB with adaptive beta based on uncertainty
    # Use a beta that scales with the overall uncertainty level
    max_sigma = np.max(sigma)
    if max_sigma > 1e-9:
        beta = 1.0 + 0.5 * (max_sigma / (np.mean(sigma) + 1e-9))
    else:
        beta = 1.0
    
    ucb = mu + beta * sigma  # For minimization, we want low mu, so UCB penalizes high uncertainty
    
    # Combine EI and UCB. Since we minimize, lower UCB is better, so we negate it.
    # Normalize both to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1e-9
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min if ucb_max > ucb_min else 1e-9
    
    # Normalize EI to [0, 1]
    ei_norm = ei / ei_max
    
    # Normalize UCB to [0, 1] (lower UCB = better, so invert)
    ucb_norm = (ucb_max - ucb) / ucb_range
    
    # Weighted combination: 60% EI, 40% UCB
    score = (0.6 * ei_norm + 0.4 * ucb_norm) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
