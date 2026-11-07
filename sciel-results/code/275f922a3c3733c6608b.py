"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: return early if we are too close to the wall-clock limit.
# The evaluator kills after 330s; we use a safety margin.
_DEADLINE = time.time() + 320.0  # 320s budget (10s margin)


def _safe_norm_cdf(x):
    """Numerically stable normal CDF."""
    return 0.5 * (1.0 + np.vectorize(np.math.erf)(x / np.sqrt(2.0)))


def _safe_norm_pdf(x):
    """Numerically stable normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
        mu: (2000,) array of GP mean predictions in objective units.
        sigma: (2000,) array of GP standard deviations in objective units.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of evaluation costs for each candidate.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a reasonable fallback: prioritize low cost and low mu
        return -mu / (cost + 1e-9)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * _safe_norm_cdf(z) + sigma * _safe_norm_pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI per unit cost
    # Use a softer cost penalty to balance exploration and exploitation
    # The cost is 1 + mean(x_scaled_to_[0,1]), so range is [1, 2]
    # We use a power penalty to emphasize cost efficiency
    score = ei / (cost ** 1.5)
    
    # Add a small exploration bonus for high uncertainty to avoid
    # completely ignoring uncertain regions
    # This helps when EI is zero (all candidates are worse than f_best)
    exploration_bonus = sigma * 0.1 / cost
    score = score + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard check
    if time.time() > _DEADLINE - 5.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
