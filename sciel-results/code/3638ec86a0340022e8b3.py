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
    # With k=1.0, the gate at max cost (1.0) is exp(-1) ~ 0.37.
    # This creates a stronger preference for cheaper evaluations,
    # which is beneficial when the total budget is limited (30).
    k = 1.0
    cost_gate = np.exp(-k * cost_norm)
    
    # Final score: EI * cost_gate
    scores = ei * cost_gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
