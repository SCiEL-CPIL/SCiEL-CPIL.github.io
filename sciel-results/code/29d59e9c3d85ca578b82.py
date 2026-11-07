"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining EI and UCB for robust optimization.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP standard deviation predictions
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost for each candidate
    
    Returns:
        (2000,) finite scores, higher = evaluate next
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-6)
        fallback = sigma_safe / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement
    diff = f_best - mu
    z = diff / sigma_safe
    z_clipped = np.clip(z, -15, 15)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    ei = diff * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # UCB component with beta=1.5
    ucb = -mu + 1.5 * sigma_safe
    
    # Blend EI and UCB: 70% EI, 30% UCB
    blended = 0.7 * ei + 0.3 * ucb
    
    # Soft cost: reduce the penalty for expensive regions
    # cost = 1 + mean(x_scaled), so cost in [1, 2]
    # Soft cost: 1 + 0.4 * (cost - 1), giving range [1, 1.4]
    cost_soft = 1.0 + 0.4 * np.clip(cost - 1.0, 0.0, 1.0)
    cost_soft = np.maximum(cost_soft, 1e-6)
    
    score = blended / cost_soft
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
