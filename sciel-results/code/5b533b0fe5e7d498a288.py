# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist


def acquisition(mu, sigma, f_best, cost):
    """
    Diversity-aware acquisition function for cost-aware Bayesian optimization.
    
    Combines cost-normalized Expected Improvement with a multiplicative diversity
    weight derived from GP uncertainty. This down-weights candidates in well-explored
    (low-sigma) regions to encourage exploration of diverse areas, while preserving
    the relative ranking of candidates within similar uncertainty levels.
    
    Args:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviation predictions
        f_best: scalar, best observed value so far
        cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
        array of shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # Diversity-aware weighting:
    # We use the GP's standard deviation as a proxy for "novelty" or "diversity".
    # High sigma indicates unexplored regions (high diversity potential).
    # Low sigma indicates well-explored regions (low diversity, high redundancy).
    
    sigma_min = np.percentile(sigma, 1)
    sigma_max = np.percentile(sigma, 99)
    sigma_range = sigma_max - sigma_min
    
    if sigma_range < 1e-6:
        # If sigma is nearly constant, diversity signal is ambiguous.
        # Use a neutral weight to avoid distorting EI ranking.
        diversity_weight = np.ones_like(sigma) * 0.5
    else:
        # Normalize sigma to [0, 1]
        sigma_norm = (sigma - sigma_min) / sigma_range
        sigma_norm = np.clip(sigma_norm, 0.0, 1.0)
        
        # Multiplicative weight:
        # Floor of 0.2 ensures we don't completely ignore exploitation in low-sigma areas.
        # Scale of 0.8 allows significant boost for high-sigma areas.
        diversity_weight = 0.2 + 0.8 * sigma_norm
    
    # Final acquisition score
    score = ei_cost * diversity_weight
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
