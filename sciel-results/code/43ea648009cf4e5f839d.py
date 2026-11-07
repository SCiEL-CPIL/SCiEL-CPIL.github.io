import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Hartmann 6D minimization.
    Combines UCB exploration with cost-normalized expected improvement.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running long, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: pure UCB-like score, no cost division to avoid issues
        z = (f_best - mu) / (sigma + 1e-9)
        # Use a simple linear combination
        return (f_best - mu) + 2.0 * sigma
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected improvement (for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z))
    # Use scipy.stats.norm for accuracy
    from scipy.stats import norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Clip EI to non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term
    # beta controls exploration; use a moderate value
    beta = 1.5
    ucb = (f_best - mu) + beta * sigma_safe
    
    # Cost-aware scoring: combine EI and UCB, normalized by cost
    # The cost term discourages expensive evaluations
    # We want higher scores to be selected, so we combine:
    # score = (alpha * EI + (1-alpha) * UCB) / cost
    # But UCB is in objective units, EI is in objective units
    # Actually, let's use a weighted combination
    
    # Normalize costs to [0, 1] range for stable scoring
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.ones_like(cost)
    
    # Blend: use EI as primary, add UCB for exploration
    # Weight based on how much we've explored (use cost as proxy)
    alpha = 0.6  # weight on EI
    
    combined = alpha * ei + (1 - alpha) * ucb
    
    # Normalize by cost (higher cost = lower score)
    score = combined / (cost_norm + 0.1)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
