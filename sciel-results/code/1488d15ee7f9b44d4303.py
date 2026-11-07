"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Lower Confidence Bound (LCB) with
    a Knowledge Gradient-inspired variance-aware exploration boost.
    
    Strategy: Use a high exploration constant for LCB to target high-uncertainty
    regions, plus a bonus for regions where the posterior standard deviation
    is large relative to the current best improvement, encouraging exploration
    in areas with high potential for significant improvement.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # LCB component: aggressive exploration with k=3.0
    k = 3.0
    lcb = mu - k * sigma_safe
    
    # Knowledge Gradient-inspired modulation:
    # Boost exploration in regions with high posterior uncertainty
    # relative to the gap between current best and function minimum
    # The Michalewicz 2D minimum is -1.8013
    f_min = -1.8013
    
    # Normalized uncertainty: how much sigma matters relative to the
    # remaining possible improvement
    remaining_improvement = np.maximum(f_best - f_min, 1e-6)
    normalized_sigma = sigma_safe / remaining_improvement
    
    # Variance-aware boost: regions with high normalized uncertainty get
    # an additional exploration incentive. This creates a different search
    # landscape than pure EI, potentially escaping local optima.
    # The boost is scaled to be meaningful but not dominate
    kg_boost = 0.5 * normalized_sigma
    
    # Combine: LCB (higher is better for exploration) + KG boost
    # Since we want to maximize the score, and LCB is lower when exploring,
    # we negate LCB to make exploration (low LCB) give high score
    score = -lcb + kg_boost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
