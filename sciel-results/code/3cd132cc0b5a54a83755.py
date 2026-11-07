"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
TIME_LIMIT = 310.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PI)
    and Expected Improvement (EI) for Bayesian optimization of Levy 2D.
    
    Uses PI as the primary signal (maximizing likelihood of any improvement)
    with a small EI component for magnitude of improvement, all normalized
    by evaluation cost.
    """
    # Deadline guard: if approaching time limit, use a very fast fallback
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT:
        # Fast fallback: simple cost-normalized negative mean
        cost_safe = np.maximum(cost, 1e-9)
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Probability of Improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine: PI is the primary signal (structural change from pure EI),
    # with a small EI component to favor larger improvements when probabilities
    # are similar. The weight on EI is kept small so PI dominates the search
    # direction, which may navigate the Levy landscape more effectively by
    # prioritizing regions with high chance of improvement over regions with
    # high expected improvement magnitude.
    score = (pi + 0.1 * ei) / cost_safe
    
    # Ensure finite output as required
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
