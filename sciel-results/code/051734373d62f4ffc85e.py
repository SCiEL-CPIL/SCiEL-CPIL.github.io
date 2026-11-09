"""Rewritten program with cost-normalized POI + deterministic regret lower bound."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining probability of improvement (POI)
    with a deterministic regret lower bound, normalized by cost.
    
    Formula: (f_best - mu + 0.5 * sigma) / cost
    
    This directly optimizes expected reduction in simple regret per unit cost.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-normalized probability of improvement combined with regret lower bound
    # (f_best - mu + 0.5 * sigma) captures:
    #   - f_best - mu: expected improvement if we evaluate at this point
    #   - 0.5 * sigma: bonus for uncertainty (encourages exploration)
    # Dividing by cost gives improvement per unit cost
    
    scores = (f_best - mu + 0.5 * sigma_safe) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
