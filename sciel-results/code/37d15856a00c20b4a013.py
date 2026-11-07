"""
Structural change: switch from a static EI+bonus hybrid to an adaptive
acquisition that uses UCB in the exploration phase (when f_best is far from
the optimum) and switches to pure EI in the exploitation phase. The switch
is triggered by progress in f_best, not by a fixed schedule.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T_START = time.time()
_TIME_BUDGET = 320.0

def _elapsed():
    return time.time() - _T_START

def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition: UCB for exploration, EI for exploitation.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
    sigma : ndarray, shape (2000,)
    f_best : float
    
    Returns
    -------
    scores : ndarray, shape (2000,)
    """
    if _elapsed() > _TIME_BUDGET:
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    n = len(mu)
    eps = 1e-9

    # Determine phase based on progress.
    # Hartmann 3d minimum is -3.86278. If f_best is still high (e.g., > -1.0),
    # we are in exploration mode. Once f_best < -1.0, switch to exploitation.
    # This is a structural switch, not a smooth blend.
    exploration_threshold = -1.0
    
    if f_best > exploration_threshold:
        # Exploration phase: UCB with gamma that decreases as we get closer
        # to the threshold
        # gamma = 1.0 when f_best is very high, approaches 0.5 near threshold
        # Use a simple linear mapping
        # When f_best = 0 (worst case), gamma = 1.0
        # When f_best = -1.0, gamma = 0.5
        # Clamp f_best to [0, -1] for the mapping
        f_clamped = np.clip(f_best, -1.0, 0.0)
        gamma = 0.5 + 0.5 * (f_clamped + 1.0)  # maps [-1, 0] to [0.5, 1.0]
        gamma = max(gamma, 0.5)
        
        # UCB for minimization: lower is better
        # We want to maximize the score, so we use -mu + gamma * sigma
        # But we need to handle the scale. Let's normalize.
        # Actually, for minimization, UCB is: -mu + gamma * sigma
        # Higher score = better
        scores = -mu + gamma * sigma
    else:
        # Exploitation phase: pure EI
        z = (f_best - mu) / (sigma + eps)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Ensure finite
    if not np.all(np.isfinite(scores)):
        finite_mask = np.isfinite(scores)
        if np.any(finite_mask):
            mean_val = np.mean(scores[finite_mask])
            scores = np.where(finite_mask, scores, mean_val)
        else:
            scores = np.zeros_like(mu)

    return scores
# EVOLVE-BLOCK-END
