"""Rewritten acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid approach: Expected Improvement (EI) for exploitation combined
    with an adaptive exploration bonus to encourage visiting distinct regions.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration bonus:
    # - Early on (high sigma), we want to explore more
    # - Later (low sigma), we want to exploit more
    # Use a scaling that decays as the best value improves
    
    # Compute a measure of how "good" the current best is
    # The Weierstrass function minimum is 0, so f_best approaching 0 means we're close
    # Use a simple heuristic: scale exploration by the current sigma level
    
    # Normalize sigma to get a relative measure
    sigma_max = np.max(sigma)
    if sigma_max > 1e-12:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = sigma
    
    # Exploration bonus: proportional to sigma, but with a cap
    # This encourages visiting high-uncertainty regions
    # Use a moderate coefficient to balance exploration and exploitation
    gamma = 0.5
    exploration_bonus = gamma * sigma
    
    # Combine EI and exploration bonus
    # The EI term drives exploitation, the exploration bonus drives exploration
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
