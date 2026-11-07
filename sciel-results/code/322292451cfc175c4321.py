"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Probability of Improvement
    and Expected Improvement, normalized by cost, with an exploration bonus
    based on predictive uncertainty.
    
    The score combines:
    1. Exploitation: how much the candidate is expected to improve over f_best
    2. Exploration: uncertainty (sigma) as a proxy for information gain
    3. Cost normalization: divide by evaluation cost
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        GP mean predictions
    sigma : array-like, shape (n,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition values (higher = more desirable to evaluate)
    """
    # Deadline guard: return a fast fallback to avoid timeout
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        score = (f_best - mu + 1.0 * sigma_safe) / (cost + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)

    # Cost should be positive; clamp to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)

    # Compute improvement: how much better than f_best
    improvement = f_best - mu  # positive if mu < f_best (better)

    # Standardized improvement for PoI
    z = improvement / sigma_safe

    # Probability of Improvement
    poi = norm.cdf(z)

    # Expected Improvement (standard form)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Combine PoI and EI: PoI captures the probability, EI captures the magnitude
    # Use a weighted combination that favors exploitation when improvement is clear
    # and exploration when uncertain
    # Weight: higher weight on EI when improvement is large, higher on PoI when uncertain
    # Simple blend: 0.5 * PoI + 0.5 * EI / sigma_safe (normalized EI)
    # But EI/sigma = z * Phi(z) + phi(z), which is bounded
    ei_normalized = np.where(sigma_safe > 1e-9, ei / sigma_safe, 0.0)
    ei_normalized = np.where(np.isfinite(ei_normalized), ei_normalized, 0.0)

    # Blend: PoI gives a probability in [0,1], EI_normalized gives expected gain in sigma units
    # Use PoI as base and add a scaled EI contribution
    # The scaling factor controls exploration vs exploitation
    exploration_weight = 0.8
    score = poi + exploration_weight * np.maximum(ei_normalized, 0.0)

    # Add a small pure exploration bonus to encourage diversity
    # This helps when all points have similar mu but different sigma
    exploration_bonus = 0.3 * sigma_safe
    score = score + exploration_bonus

    # Normalize by cost (cost-aware)
    score = score / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
