"""Deterministic acquisition function: POI weighted by dynamic exploration, normalized by cost."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    normalized by cost. EI naturally balances exploitation (improvement)
    and exploration (uncertainty) and is weighted by the probability of
    improvement (POI) in its derivation.
    
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
    
    # Probability Density Function at z
    pdf_z = norm.pdf(z)
    
    # Expected Improvement:
    # EI = (f_best - mu) * P(f(x) < f_best) + sigma * phi(z)
    # This term is 0 when f(x) is likely to be worse than f_best.
    ei = (f_best - mu) * poi + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Normalize by cost to prioritize low-cost evaluations
    # Add a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
