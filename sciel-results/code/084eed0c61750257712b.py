"""Rewritten program with global budget-aware scaling strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using global budget-aware scaling.
    
    Instead of point-wise division by cost, we compute the expected improvement
    and then apply a global penalty based on the ratio of each candidate's cost
    to the mean cost of the batch. This shifts the exploration-exploitation
    balance based on remaining budget rather than local cost ratios.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        Mean of the GP posterior in objective units.
    sigma : ndarray of shape (2000,)
        Standard deviation of the GP posterior in objective units.
    f_best : float
        Best observed value so far (lowest).
    cost : ndarray of shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Compute expected improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Handle cases where improvement is negative (mu > f_best)
    # In those cases, EI should be 0 (or very small)
    ei = np.zeros_like(mu)
    positive_mask = improvement > 0
    if np.any(positive_mask):
        ei[positive_mask] = (
            improvement[positive_mask] * norm.cdf(z[positive_mask]) +
            sigma_safe[positive_mask] * norm.pdf(z[positive_mask])
        )
    
    # Global budget-aware scaling:
    # Use the minimum cost in the batch as the baseline for the penalty.
    # This ensures the cheapest candidate has zero penalty, and more expensive ones
    # are penalized relative to the best possible cost choice in this batch.
    min_cost = np.min(cost)
    if min_cost < eps:
        min_cost = eps
    
    # Cost ratio relative to the minimum cost
    cost_ratio = cost / min_cost
    
    # Global scaling: normalize EI by its max (if positive) to get a [0, 1]-ish scale
    ei_max = np.max(ei)
    if ei_max > eps:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = ei
    
    # Apply cost penalty: subtract a term proportional to (cost_ratio - 1)
    # The penalty is 0 for the cheapest candidate and increases linearly.
    # Coefficient of 1.0 means a candidate with 2x the minimum cost gets a penalty of 1.0,
    # which is significant compared to the normalized EI range [0, 1].
    penalty_coeff = 1.0
    penalty = penalty_coeff * (cost_ratio - 1.0)
    
    # Final score: normalized EI minus penalty
    scores = ei_normalized - penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
