"""Deterministic acquisition function: POI weighted by dynamic exploration, normalized by cost."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    weighted by a dynamic exploration factor derived from global max uncertainty.
    
    Parameters:
    -----------
    mu : array_like, shape (N,)
        Mean predictions from GP in objective units.
    sigma : array_like, shape (N,)
        Standard deviation predictions from GP in objective units.
    f_best : float
        Best observed function value.
    cost : array_like, shape (N,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores (higher = more promising).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    poi = norm.cdf(z)
    
    # Expected Improvement components
    # Improvement potential: how much better can we get?
    # Clip to 0 to ensure we don't reward points predicted to be worse than f_best
    improvement = np.maximum(f_best - mu, 0.0)
    
    # Exploration term: UCB-like bonus
    # We use a fixed kappa for now, but scale it dynamically if possible.
    # Since we don't have observed data scale, we rely on the fact that
    # multiplying by POI (which is high only when improvement is likely)
    # naturally balances this.
    kappa = 2.0
    exploration_bonus = kappa * sigma_safe
    
    # Combined score: POI-gated (Improvement + Exploration)
    # This ensures that areas with low probability of improvement (high mu)
    # get a low score, even if uncertainty is high.
    # This fixes the issue of exploring cheap but bad regions.
    combined = poi * (improvement + exploration_bonus)
    
    # Normalize by cost to prioritize low-cost evaluations
    # Add a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
