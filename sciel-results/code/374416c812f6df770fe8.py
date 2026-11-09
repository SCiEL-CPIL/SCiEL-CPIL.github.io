"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using multiplicative cost gating.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (clip small negative values from numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Multiplicative cost gate with normalized cost
    # Cost ranges from 1 to 2. Normalize to [0, 1] so that the cheapest points
    # are not penalized (gate=1 for min cost).
    # This allows a stronger relative penalty for expensive points.
    cost_norm = cost - 1.0
    
    # Calibrate k for the normalized cost range [0, 1].
    # Increased k to 3.0 to strongly suppress high-cost points.
    # Gate at max cost (1.0) is exp(-3) ~ 0.05.
    # This enforces a strict cost-awareness: expensive points are only chosen
    # if their expected improvement is exceptionally high (>20x that of a cheap point).
    # This helps conserve the limited budget (30) for more evaluations.
    k = 3.0
    cost_gate = np.exp(-k * cost_norm)
    
    # Final score: EI * cost_gate
    scores = ei * cost_gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
