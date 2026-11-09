"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) with exploration bonus.
    
    Uses PI = Phi((f_best - mu) / sigma) normalized by sqrt(cost).
    Adds a small sigma-based term to enhance exploration in multimodal landscapes.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = better)
    """
    # Ensure sigma and cost are positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # Using erf for numerical stability and speed: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    pi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # Base score: PI normalized by sqrt(cost)
    scores = pi / np.sqrt(cost_safe)
    
    # Add a small exploration term proportional to uncertainty (sigma)
    # This helps escape local minima in Ackley function
    exploration_bonus = 0.1 * sigma_safe / np.sqrt(cost_safe)
    
    scores += exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
