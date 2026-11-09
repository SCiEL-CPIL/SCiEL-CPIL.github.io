# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = None
DEADLINE_FRACTION = 0.92  # use 92% of 300s to be safe


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (EI) and UCB
    with a time-based schedule that shifts from exploration to exploitation.
    
    Returns 2000 finite scores; higher means evaluate next.
    """
    global _start_time
    
    # Ensure timer is initialized
    if _start_time is None:
        _start_time = time.perf_counter()
    
    elapsed = _get_elapsed()
    
    # Deadline guard: if we're running out of time, fall back to simple EI
    # to ensure we return quickly
    if elapsed > DEADLINE_FRACTION * 300.0:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute EI scores
    z_ei = (f_best - mu) / (sigma + 1e-9)
    ei_scores = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Compute UCB scores
    # UCB = -mu + beta * sigma
    # We need to choose beta. A common approach is to make UCB comparable in scale to EI.
    # Since mu and sigma are in objective units, and we're minimizing,
    # UCB score = -mu + beta * sigma (higher is better)
    
    # Adaptive beta: start with higher exploration, decay over time
    # Assuming ~30 evaluations total, and each step takes some time
    # Let's estimate progress: if total budget is 300s and we have 30 evals,
    # roughly 10s per eval. But this is rough.
    # Use elapsed time as a proxy: at start, beta is high; near end, beta is low.
    
    # Estimate total time: 300s
    # Estimate progress: elapsed / 300
    progress = min(elapsed / 300.0, 1.0)
    
    # Beta schedule: start at 3.0, decay to 0.5
    # beta = beta_max - (beta_max - beta_min) * progress
    beta_max = 3.0
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    ucb_scores = -mu + beta * sigma
    
    # Normalize both to [0, 1] range for blending
    # Use min-max normalization with small epsilon for stability
    
    def _normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng
    
    ei_norm = _normalize(ei_scores)
    ucb_norm = _normalize(ucb_scores)
    
    # Blend weight: start with more UCB (exploration), shift to EI (exploitation)
    # At start: w_ei = 0.3, w_ucb = 0.7
    # At end:   w_ei = 0.8, w_ucb = 0.2
    w_ei = 0.3 + 0.5 * progress
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
