import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimizing the objective.

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
        score = f_best - mu
        return np.clip(score, -1e12, 1e12)

    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Standardized improvement: (f_best - mu) / sigma
    # For minimization, improvement = f_best - f(x)
    # EI = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in Phi/phi for extreme values
    z_clipped = np.clip(z, -10, 10)
    
    Phi = norm.cdf(z_clipped)
    phi = norm.pdf(z_clipped)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI is non-negative by construction, but clip for safety
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.clip(EI, 0, 1e12)
    
    # Guarantee exactly `n` finite entries (defensive).
    if EI.shape[0] != n:
        EI = np.resize(EI, n)
    
    return EI
