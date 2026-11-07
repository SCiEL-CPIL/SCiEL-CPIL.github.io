"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
import time as _time

_DEADLINE_S = 280.0
_START = None
_CALL_COUNT = 0


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def _ei_fast(mu, sigma, f_best):
    """
    Fast, stable Gaussian Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma, Phi is standard normal CDF, phi is PDF.
    """
    eps = 1e-15
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    cdf = 0.5 * erfc(-z / np.sqrt(2.0))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = d * cdf + sigma_safe * pdf
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    ei = np.maximum(ei, 0.0)
    return ei


def acquisition(mu, sigma, f_best, cost):
    global _START, _CALL_COUNT
    if _START is None:
        _START = _now()
    _CALL_COUNT += 1

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    ei = _ei_fast(mu, sigma, f_best)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # Adaptive cost exponent: stricter penalty late in the run
    mean_sigma = np.mean(sigma)
    sigma_norm = mean_sigma / (mean_sigma + 0.05)
    alpha_cost = 0.35 + 0.9 * (1.0 - sigma_norm)
    alpha_cost = np.clip(alpha_cost, 0.3, 1.4)

    cost_safe = np.maximum(cost, 1.0)
    cost_factor = cost_safe ** alpha_cost

    ei_cost = ei / cost_factor

    # Exploration boost scaled by cost
    explore_boost = 0.015 * sigma / cost_safe

    score = ei_cost + explore_boost
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
