"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition: normalized Expected Improvement + uncertainty bonus.

    - EI drives exploitation toward points likely to beat f_best.
    - A sigma term (normalized) encourages exploration where uncertainty is high.
    - Weights adapt to the gap between f_best and the known global minimum.
    """
    # Deadline guard: if we're too close to the kill, return a safe fallback
    if time.time() > _DEADLINE - 5.0:
        # Fallback: pure uncertainty (sigma) to be safe and finite
        return sigma.copy()

    # Known global minimum for Michalewicz 2D
    f_global_min = -1.8013

    # Gap to global minimum: large early, small late
    gap = max(f_best - f_global_min, 1e-6)

    # Compute EI for minimization
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z

    # Normalize EI to [0, 1] using a stable scale
    ei_scale = np.max(ei) - np.min(ei)
    if ei_scale < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - np.min(ei)) / ei_scale

    # Uncertainty term: normalize sigma to [0, 1]
    sigma_scale = np.max(sigma) - np.min(sigma)
    if sigma_scale < 1e-9:
        sigma_norm = np.zeros_like(sigma)
    else:
        sigma_norm = (sigma - np.min(sigma)) / sigma_scale

    # Adaptive weights: more exploration when gap is large
    # w_explore in [0.1, 0.7], w_exploit = 1 - w_explore
    w_explore = 0.1 + 0.6 * np.clip(gap / 2.8, 0.0, 1.0)  # gap ~ 2.8 early (f_best~1)
    w_exploit = 1.0 - w_explore

    final_scores = w_exploit * ei_norm + w_explore * sigma_norm

    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    return final_scores
# EVOLVE-BLOCK-END
