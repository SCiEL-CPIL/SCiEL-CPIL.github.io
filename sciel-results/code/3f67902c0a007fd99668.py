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
    
    # Multiplicative cost gate: exp(-k * cost)
    # Calibrate k to the expected EI scale
    # The cosine8 function has a range of roughly [-0.8, 1.0], so typical EI values
    # are on the order of 0.1 to 0.5 in the early/mid stages of optimization.
    # Costs range from 1 to 2 (since mean(x) in [0,1] means cost in [1, 2]).
    # We want high-cost points to be exponentially suppressed unless EI is large.
    # Let's set k such that at cost=2 (max), the gate is exp(-2k).
    # If we want the gate to reduce score by ~50% at max cost when EI is moderate,
    # we need exp(-2k) ≈ 0.5 => k ≈ 0.35
    # But we also want to allow high-EI points to overcome the cost penalty.
    # A reasonable k is around 0.3-0.5. Let's use 0.4.
    k = 0.4
    cost_gate = np.exp(-k * cost)
    
    # Final score: EI * cost_gate
    scores = ei * cost_gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
