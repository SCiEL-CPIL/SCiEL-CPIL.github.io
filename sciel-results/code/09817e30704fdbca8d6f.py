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
    Combines Thompson Sampling (randomized posterior sample) with
    sigma-inflated Expected Improvement and an adaptive LCB component.
    Early on, Thompson Sampling dominates for broad exploration;
    later, EI and LCB take over for exploitation.
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    t = _CALL_COUNT

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)

    # --- Thompson Sampling component ---
    # Draw a single sample from the posterior predictive: mu - sigma * N(0,1)
    # This naturally balances exploration (high sigma regions get large swings)
    # and exploitation (low mu regions tend to score higher when sampled).
    # Use a fixed seed per call index for reproducibility within a run.
    rng = np.random.default_rng(42 + t)
    ts_sample = mu - sigma * rng.standard_normal(len(mu))

    # --- Sigma-inflated Expected Improvement ---
    inflation = 2.0
    sigma_eff = sigma * inflation
    sigma_safe = np.where(sigma_eff > 1e-9, sigma_eff, 1e-9)

    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Adaptive LCB component ---
    # Beta decays over time: high exploration early, tight exploitation late
    beta_start = 3.0
    beta_decay = 0.10
    beta = beta_start / (1.0 + beta_decay * t)
    lcb_score = -mu + beta * sigma_safe

    # --- Normalize all three components to [0, 1] ---
    def _normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        rng_ = mx - mn
        if rng_ < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / rng_

    ts_norm = _normalize(ts_sample)
    ei_norm = _normalize(ei)
    lcb_norm = _normalize(lcb_score)

    # --- Adaptive weighting ---
    # Phase 1 (t < 8): Thompson Sampling dominates (broad exploration)
    # Phase 2 (8 <= t < 18): blend TS with EI/LCB
    # Phase 3 (t >= 18): EI and LCB dominate (exploitation)
    #
    # w_ts decays: strong early, weak late
    w_ts = np.exp(-t / 8.0)
    # w_ei and w_lcb split the remaining weight, with LCB stronger early
    w_lcb = (1.0 - w_ts) * np.exp(-t / 15.0)
    w_ei = (1.0 - w_ts) - w_lcb

    # Ensure weights are non-negative and sum to 1
    w_ts = max(w_ts, 0.0)
    w_lcb = max(w_lcb, 0.0)
    w_ei = max(w_ei, 0.0)
    w_sum = w_ts + w_lcb + w_ei
    if w_sum < 1e-12:
        w_ts, w_lcb, w_ei = 1.0, 0.0, 0.0
    else:
        w_ts /= w_sum
        w_lcb /= w_sum
        w_ei /= w_sum

    # --- Blend ---
    score = w_ts * ts_norm + w_ei * ei_norm + w_lcb * lcb_norm

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
