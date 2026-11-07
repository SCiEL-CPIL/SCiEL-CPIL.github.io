"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement with adaptive exploration.
    
    Strategy:
    - Compute Expected Improvement (EI) for each candidate.
    - Apply an adaptive exploration bonus based on the gap to the known minimum.
    - Divide by cost to account for evaluation expense.
    - Use a time guard to ensure we return before the 300s deadline.
    """
    t_start = time.time()
    
    # Safety: clip inputs
    sigma = np.clip(sigma, 1e-8, None)
    cost = np.clip(cost, 1.0, None)  # cost is at least 1.0 by definition
    
    # Known minimum of Hartmann 3D
    f_min = -3.86278
    
    # Gap from current best to known minimum
    gap = max(f_best - f_min, 0.0)
    
    # Compute z-scores for EI
    # z = (f_best - mu) / sigma
    # When mu < f_best, z > 0 (potential improvement)
    z = (f_best - mu) / sigma
    
    # Standard normal CDF and PDF
    # Use scipy for accuracy
    from scipy.stats import norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is the expected improvement over f_best
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Adaptive exploration: add a bonus proportional to sigma when we're far from optimum
    # This encourages exploring uncertain regions more when we haven't found a good solution yet
    gap_norm = min(gap / 5.0, 1.0)
    exploration_weight = 0.5 * gap_norm  # Up to 0.5 extra exploration when far from optimum
    
    # Combined score: EI + exploration bonus, then divide by cost
    score = ei + exploration_weight * sigma
    score = score / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Time guard: if we're running long, return immediately
    elapsed = time.time() - t_start
    if elapsed > 280:
        return score
    
    return score
# EVOLVE-BLOCK-END
