"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a modified Expected Improvement
    with a quadratic penalty for high cost, combined with an uncertainty
    bonus to encourage exploration in early stages.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean predictions from the GP.
    sigma : array-like, shape (n,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute the z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment:
    # We want to penalize high-cost points more heavily.
    # A simple approach is to divide by cost, but this can be too aggressive.
    # Instead, we use a power-law penalty: ei / (cost ** alpha)
    # with alpha > 1 to penalize high costs more.
    
    # Also, add an uncertainty bonus to encourage exploration.
    # The bonus is proportional to sigma but scaled down as we get more confident.
    # We use a simple heuristic: bonus = beta * sigma / (1 + f_best)
    # where beta controls the strength of exploration.
    
    # Let's use a more principled approach:
    # 1. Compute EI per unit cost
    # 2. Add a small exploration bonus based on uncertainty
    
    alpha = 1.5  # Cost penalty exponent
    beta = 0.1   # Exploration bonus coefficient
    
    # EI per unit cost with power-law penalty
    ei_cost_aware = ei / (cost_safe ** alpha)
    
    # Exploration bonus: reward points with high uncertainty
    # Scale the bonus so it's meaningful relative to EI
    exploration_bonus = beta * sigma_safe / (1.0 + np.abs(f_best))
    
    # Combine
    scores = ei_cost_aware + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
