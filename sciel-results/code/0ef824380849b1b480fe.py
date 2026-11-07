import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin well below the 330 s wall-clock kill

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition for minimizing the objective.
    
    EI(x) = E[max(f_best - f(x), 0)]
         = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    This directly maximizes the expected gain over the current best, providing a
    theoretically grounded balance between exploration and exploitation.
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

    # Compute EI for minimization
    # improvement = f_best - mu (positive when prediction is better than best)
    improvement = f_best - mu
    z = improvement / sigma_safe

    # EI = improvement * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)

    ei = improvement * Phi_z + sigma_safe * phi_z

    # Keep the output strictly finite and bounded.
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.clip(ei, -1e12, 1e12)

    # Guarantee exactly `n` finite entries (defensive).
    if ei.shape[0] != n:
        ei = np.resize(ei, n)

    return ei
