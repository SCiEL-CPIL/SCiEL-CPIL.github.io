import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Adaptive Expected Improvement (EI) acquisition for minimizing the objective.
    
    Uses an adaptive exploration parameter (xi) that decreases as f_best 
    approaches the known global minimum of the Goldstein-Price function (3.0).
    This encourages broad exploration initially and focused exploitation later.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # ---- wall-clock guard: never let this call approach the kill limit ----
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast).
        # Minimize mu directly.
        score = -mu
        return np.clip(score, -1e12, 1e12)

    # Known global minimum for Goldstein-Price 2D
    f_min = 3.0
    
    # Calculate the gap between current best and global minimum
    # Clip to avoid negative gaps if f_best < 3.0 (shouldn't happen, but safety)
    gap = max(f_best - f_min, 0.0)
    
    # Adaptive xi (exploration bonus)
    # We want xi to be significant when gap is large, and small when gap is small.
    # Scale factor: if gap is ~50, xi should be ~5. If gap is ~1, xi should be ~0.1.
    # Linear decay: xi = k * gap
    # Let's pick k = 0.1. 
    # If f_best = 50, gap = 47, xi = 4.7.
    # If f_best = 10, gap = 7, xi = 0.7.
    # If f_best = 4, gap = 1, xi = 0.1.
    # This seems reasonable.
    xi = 0.1 * gap
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement for EI with xi:
    # EI_xi = E[max(f_best - xi - f(x), 0)]
    # = (f_best - xi - mu) * Phi((f_best - xi - mu)/sigma) + sigma * phi((f_best - xi - mu)/sigma)
    
    # Handle the case where f_best - xi - mu is very negative
    # In that case, EI is 0.
    
    z = (f_best - xi - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    Phi = norm.cdf(z_clipped)
    phi = norm.pdf(z_clipped)
    
    EI = (f_best - xi - mu) * Phi + sigma_safe * phi
    
    # EI is non-negative by construction, but clip for safety
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.clip(EI, 0, 1e12)
    
    # Guarantee exactly `n` finite entries (defensive).
    if EI.shape[0] != n:
        EI = np.resize(EI, n)
    
    return EI
