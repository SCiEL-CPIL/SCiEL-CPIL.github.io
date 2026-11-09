"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Probability of Improvement (PI) per sqrt(cost).
    
    PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    We normalize by cost^0.5 to balance exploration efficiency.
    
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
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Compute z-score for probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Normalize by cost to the power of 0.5
    # cost is in [1, 2] since it's 1 + mean(x_scaled)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Use a cost-weighted PI with slight exploration bias
    # The cost term encourages cheaper evaluations
    cost_norm = np.sqrt(cost_safe)
    
    # Add a small exploration bonus proportional to sigma to encourage
    # exploring uncertain regions, but keep it small to not dominate
    exploration_bonus = 0.05 * sigma_safe / np.max(sigma_safe)
    
    scores = pi / cost_norm + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
