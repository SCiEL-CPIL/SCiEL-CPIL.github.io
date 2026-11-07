"""Bayesian optimization acquisition for Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_obs_count = 5
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) and 
    Lower Confidence Bound (LCB) for minimization.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points (in objective units).
    f_best : float
        Best observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    global _obs_count
    t = _obs_count
    _obs_count += 1

    # Deadline guard: if we're close to the limit, heavily favor exploitation
    # to ensure we finish and potentially squeeze out a final improvement.
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Pure exploitation: pick the point with the lowest mean
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Dynamic exploration parameter (kappa)
    # Starts high for exploration, decays over time to favor exploitation.
    # Beale function is narrow, so we need enough exploration early to find the valley.
    kappa = 2.0 + 8.0 * np.exp(-t / 10.0)

    # 1. Expected Improvement (EI)
    # EI measures the expected amount by which f(x) improves upon f_best.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # 2. Lower Confidence Bound (LCB) for minimization
    # We want to minimize f(x). The lower bound is mu - kappa * sigma.
    # To select the best point (maximize score), we use -LCB = -mu + kappa * sigma.
    # This rewards low mean (exploitation) and high variance (exploration).
    lcb_score = -mu + kappa * sigma

    # Helper function to normalize an array to [0, 1]
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / rng

    ei_norm = normalize(ei)
    lcb_norm = normalize(lcb_score)

    # Combine: Balanced mix of probabilistic EI and heuristic LCB
    # EI is robust, LCB helps navigate the "valley" structure by explicitly
    # encouraging exploration in uncertain regions with low means.
    scores = 0.5 * ei_norm + 0.5 * lcb_norm

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
