"""Cost-aware acquisition for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
    Uses a diversity-promoting strategy:
    - Expected Improvement (EI) as the primary exploitation term
    - Penalize candidates that are spatially close to already-evaluated points
      by using the GP's predictive variance (sigma) as a proxy for information gain
    - Normalize by cost to get expected improvement per unit cost
    
    The key idea: candidates with high sigma (far from evaluated points) get an
    exploration bonus, while candidates with low sigma but high EI get an
    exploitation bonus. We combine these to encourage domain coverage.
    """
    # Safety margin for time
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Diversity term: use sigma as a proxy for novelty/information gain
    # Higher sigma means the point is far from evaluated data, so it's more "novel"
    # We want to balance exploitation (EI) with exploration (sigma)
    
    # Normalize EI to [0, 1] range for combining
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.ones(n)
    
    # Normalize sigma to [0, 1] range
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.ones(n)
    
    # Combine: weighted sum of normalized EI and sigma
    # Weight exploration more heavily early on, but we don't know the iteration
    # Use a fixed weight that balances both
    w_explore = 0.3  # 30% weight to exploration (sigma)
    w_exploit = 1.0 - w_explore  # 70% weight to exploitation (EI)
    
    combined = w_exploit * ei_norm + w_explore * sigma_norm
    
    # Normalize by cost (cost-aware)
    cost_safe = np.maximum(cost, 1e-9)
    score = combined / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time check: if we're running low on time, return immediately
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
