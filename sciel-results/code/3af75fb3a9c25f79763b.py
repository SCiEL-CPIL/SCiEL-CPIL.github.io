"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time as _time

_DEADLINE_S = 240.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def acquisition(mu, sigma, f_best, cost):
    global _START
    if _START is None:
        _START = _now()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite inputs.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    n = mu.shape[0]

    # Deadline guard: if we are too close to the kill, return a cheap
    # exploitation-only score so we never blow the budget mid-call.
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        z = np.clip(z, -10.0, 10.0)
        return np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)

    eps = 1e-9
    sigma_safe = sigma + eps

    # --- Expected improvement (EI) for minimization ---
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)  # avoid overflow/underflow in cdf/pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(sigma > 0, ei, np.maximum(f_best - mu, 0.0))
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # --- Cost-aware scaling ---
    # cost = 1 + mean(x_scaled) ∈ [1, 2].
    # Normalize to [0.5, 1.0] so the cost factor is a gentle 2x max preference
    # for the cheapest points, applied as a power for tunable strength.
    cost_norm = 0.5 + 0.5 * (cost - 1.0)  # maps [1,2] -> [0.5,1.0]
    cost_norm = np.clip(cost_norm, 0.5, 1.0)
    beta = 1.0  # cost exponent: 1.0 gives ei / cost_norm, up to 2x preference
    cost_aware = ei / (cost_norm ** beta)

    # --- Confidence-adaptive exploitation ---
    # When sigma is tiny relative to the problem scale, the model is confident
    # and we should exploit the predicted minimum rather than diffuse budget.
    # Use a smooth sigmoid in log-sigma to avoid the hard discontinuity of
    # exp(-sigma/1e-3).
    # sigma_ref: a scale at which we consider the model "uncertain".
    # For these test functions (range ~10, values ~0-100), sigma ~ 1.0 is
    # a reasonable reference.
    sigma_ref = 0.5
    # logit: large positive when sigma << sigma_ref, large negative when sigma >> sigma_ref
    logit = np.log(sigma_safe / sigma_ref)
    # Smooth sigmoid: ~1 when sigma << sigma_ref, ~0 when sigma >> sigma_ref
    conf = 1.0 / (1.0 + np.exp(logit))
    conf = np.clip(conf, 0.0, 1.0)

    # Exploitation score: predicted improvement over f_best, cost-normalized.
    exploit = np.maximum(f_best - mu, 0.0) / (cost_norm ** beta)

    # Blend: small weight toward exploitation when confident.
    # The blend weight scales with confidence so it's 0 when uncertain
    # and up to 0.15 when very confident.
    blend_weight = 0.15 * conf
    score = (1.0 - blend_weight) * cost_aware + blend_weight * exploit

    # Ensure finiteness.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
