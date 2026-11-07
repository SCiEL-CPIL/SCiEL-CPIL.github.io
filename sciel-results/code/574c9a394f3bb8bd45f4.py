"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf, exp
import time

# Track elapsed time for deadline guard
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Cost-Weighted Expected Improvement (CW-EI)
    and Cost-Weighted Uncertainty (CW-U).
    
    Strategy:
    - Compute Expected Improvement (EI) using fast vectorized operations.
    - Compute a cost penalty using a power law: 1 / cost^p.
    - The score is a weighted sum of EI/cost^p and sigma/cost^p.
    - This structural change moves away from normalized blending to a direct
      cost-efficiency metric, which is more stable and responsive to the
      budget constraint.
    """
    # Deadline guard: if we're running long, return a safe fallback
    elapsed = _get_elapsed()
    if elapsed > 290:
        return np.zeros(len(mu))
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Fast Expected Improvement calculation
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Use erf for cdf: cdf(z) = 0.5 * (1 + erf(z / sqrt(2)))
    sqrt_2 = np.sqrt(2.0)
    cdf_z = 0.5 * (1.0 + erf(z / sqrt_2))
    
    # pdf(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    pdf_z = exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    # EI should be non-negative, but clip for safety
    ei = np.maximum(ei, 0.0)
    
    # Cost processing
    # Cost is in [1, 2]
    cost_safe = np.maximum(cost, 1.0)
    
    # Structural Change: Power-law cost penalty
    # Using p=1.5 balances exploration in expensive regions (if EI is high)
    # with the need to conserve budget.
    # Current program used exponential decay which was very aggressive.
    # This power law is less aggressive, allowing the GP to guide search
    # into expensive regions if the potential gain (EI) is high enough.
    p = 1.5
    cost_weight = 1.0 / (cost_safe ** p)
    
    # Score components
    # 1. Exploitation: EI per cost^p
    score_ei = ei * cost_weight
    
    # 2. Exploration: Uncertainty per cost^p
    # Weight is lower than EI to prioritize known good regions,
    # but ensures we don't get stuck in local minima in cheap regions
    # if the global min is elsewhere.
    score_explore = sigma_safe * cost_weight * 0.5
    
    # Combine
    scores = score_ei + score_explore
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
