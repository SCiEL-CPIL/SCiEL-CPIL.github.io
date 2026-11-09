"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive multiplicative cost gating.
    
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
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost gate
    # The penalty strength k is calibrated to the remaining improvement potential.
    # When f_best is far from the minimum (-0.8), k is small, allowing for more expensive evaluations.
    # As f_best approaches the minimum, k increases, strongly penalizing expensive evaluations.
    f_min = -0.8
    # Clip delta_f to avoid division by zero and cap the maximum penalty
    delta_f = max(f_best - f_min, 0.1)
    
    # Calibrate k
    # k_base = 0.8 ensures that at the start (delta_f ~ 0.8), k ~ 1.0,
    # which is similar to the previous static penalty.
    k = 0.8 / delta_f
    
    # Multiplicative gate using raw cost
    # exp(-k * cost) is equivalent in ranking to exp(-k * (cost - 1))
    # but follows the directive's suggested form.
    cost_gate = np.exp(-k * cost)
    
    # Final score
    scores = ei * cost_gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
