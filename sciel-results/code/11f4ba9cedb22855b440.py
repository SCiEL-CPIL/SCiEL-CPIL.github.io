# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_TOTAL_BUDGET = 330.0
_SAFETY_MARGIN = 5.0
_CALL_COUNT = 0


def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _deadline_check(safety_margin=_SAFETY_MARGIN):
    """Return True if we are safe to continue."""
    return _get_elapsed() < _TOTAL_BUDGET - safety_margin


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of beale 2d.
    Uses sigma-inflated Expected Improvement with an adaptive LCB bonus.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    t = _CALL_COUNT

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)

    # Static sigma inflation: strong exploration to locate narrow Beale minimum
    inflation = 2.0
    sigma_eff = sigma * inflation
    sigma_safe = np.where(sigma_eff > 1e-9, sigma_eff, 1e-9)

    # Compute sigma-inflated Expected Improvement (raw, in objective units)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive LCB bonus: exploration weight decays gently over time
    beta_start = 2.5
    beta_decay = 0.05
    beta = beta_start / (1.0 + beta_decay * t)

    # LCB bonus (added on top of EI, not blended via normalization)
    lcb_bonus = beta * sigma

    # Combine: EI is the primary signal, LCB bonus adds exploration
    score = ei + lcb_bonus

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
