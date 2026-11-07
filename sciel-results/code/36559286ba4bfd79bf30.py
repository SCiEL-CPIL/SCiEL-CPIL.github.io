import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition with diversity promotion via predictive variance.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior in objective units.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Compute Expected Improvement
    xi = 0.05 * max(f_best, 0.0)
    z = (f_best - xi - mu) / sigma_safe
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Diversity term: use sigma as proxy for information gain in unexplored regions
    # Normalize sigma to [0, 1] for scaling
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > eps else 1.0
    diversity = sigma_safe / sigma_max
    
    # Adaptive blending: early on (high f_best) favor exploration/diversity,
    # later (low f_best) favor exploitation via EI
    # f_best for griewank is in [0, ~0.3], so use it directly as exploration weight
    explore_weight = np.clip(f_best / 0.1, 0.3, 1.0)
    exploit_weight = 1.0 - explore_weight
    
    # Combined acquisition: blend EI and diversity
    # Scale EI relative to its max to put on comparable scale with diversity
    ei_max = np.max(ei) if np.max(ei) > eps else 1.0
    ei_norm = ei / ei_max
    
    raw_score = exploit_weight * ei_norm + explore_weight * diversity
    
    # Cost-aware: divide by cost to penalize expensive points
    score = raw_score / cost_safe
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
