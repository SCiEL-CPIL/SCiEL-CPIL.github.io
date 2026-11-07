"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # 300s limit, keep 15s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a cost-normalized Lower Confidence Bound (LCB) with dynamic
    exploration scaling based on progress toward the known minimum.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at 2000 candidate points
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points
    f_best : float
        Best observed function value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return np.ones(2000) / cost_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Progress toward the known minimum of cosine8 (-0.8)
    # Initial f_best is typically around 0 to 0.2 for uniform random starts
    # When f_best is far from -0.8, explore more; when close, exploit more
    f_min_known = -0.8
    f_initial_estimate = 0.1  # Typical initial best from uniform random
    
    progress = np.clip((f_initial_estimate - f_best) / (f_initial_estimate - f_min_known), 0.0, 1.0)
    
    # Dynamic exploration parameter:
    # Starts high (exploration) and decreases (exploitation) as we get closer to minimum
    # beta = beta_max * (1 - progress)^1.5 + beta_min
    beta_max = 2.0
    beta_min = 0.3
    beta = beta_max * (1.0 - progress) ** 1.5 + beta_min
    
    # Lower Confidence Bound for minimization:
    # LCB = mu - beta * sigma
    # We want to maximize the negative of LCB (or minimize LCB)
    # For scoring (higher is better): score = -LCB = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-aware normalization:
    # Cost is in [1, 2] (1 + mean(x_scaled_to_01))
    # We divide by cost to normalize, but also add a small cost penalty
    # to avoid selecting very expensive points even if they have good LCB
    
    # Primary term: LCB score per unit cost
    cost_normalized_score = lcb_score / cost_safe
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps when the GP is very uncertain and we need to explore
    # But scale it down as we progress (exploit more later)
    exploration_bonus = 0.05 * (1.0 - progress) * sigma_safe / (sigma_safe + 0.5)
    
    # Combine: cost-normalized LCB + small exploration bonus
    scores = cost_normalized_score + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
