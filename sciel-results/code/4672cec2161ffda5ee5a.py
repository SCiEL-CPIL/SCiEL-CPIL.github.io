import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Dynamic Lower Confidence Bound (LCB) acquisition for minimizing the objective.
    
    The score is defined as: Score = beta * sigma - mu
    Higher scores indicate candidates that are more likely to yield a new minimum,
    balancing exploitation (low mu) and exploration (high sigma).
    
    beta is dynamically adjusted based on the current best observed value f_best.
    When f_best is large (far from the known minimum of 3.0), beta is larger to
    encourage exploration. When f_best is small (close to the minimum), beta is
    smaller to encourage exploitation.
    
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
        # If time is running out, just pick the lowest mean.
        score = -mu
        return np.clip(score, -1e12, 1e12)

    # Handle potential numerical issues in sigma
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Dynamic beta calculation
    # Goal: Larger beta when f_best is large (explore), smaller beta when f_best is small (exploit)
    # Reference minimum for Goldstein-Price is 3.0
    # f_best is always >= 3.0 in practice if the function is correct, but let's be safe.
    
    # Normalization factor to keep beta in a reasonable range [2.0, 3.0]
    # As f_best -> 3, exp(-0.1 * (f_best - 3)) -> 1. Term -> 1. Beta -> 2.0 + 1.0 * (1 - 1) = 2.0? 
    # Let's refine the formula:
    # beta = 2.0 + 1.0 * (1 - exp(-0.1 * max(0, f_best - 3.0)))
    
    # If f_best = 3.0, max(0, 0) = 0, exp(0) = 1, beta = 2.0 + 1.0 * 0 = 2.0.
    # If f_best = 13.0, max(0, 10) = 10, exp(-1) = 0.36, beta = 2.0 + 1.0 * (1 - 0.36) = 2.64.
    # If f_best = 33.0, max(0, 30) = 30, exp(-3) = 0.05, beta = 2.0 + 1.0 * (1 - 0.05) = 2.95.
    # If f_best = 103.0, beta ~ 3.0.
    
    diff = max(0.0, f_best - 3.0)
    exploration_factor = 1.0 - np.exp(-0.1 * diff)
    beta = 2.0 + 1.0 * exploration_factor
    
    # LCB score for minimization: minimize (mu - beta * sigma)
    # Since we return scores where higher is better:
    # Score = -(mu - beta * sigma) = beta * sigma - mu
    
    score = beta * sigma_safe - mu

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip to prevent overflow issues in downstream comparisons if any
    score = np.clip(score, -1e12, 1e12)
    
    # Guarantee exactly `n` finite entries (defensive).
    if score.shape[0] != n:
        score = np.resize(score, n)
        
    return score
