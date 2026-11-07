"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
from scipy.stats import norm
import time as _time

_DEADLINE_S = 285.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def _ei_fast(mu, sigma, f_best):
    """Fast Expected Improvement for minimization using erfc for stability."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    
    # norm.cdf(z) = 0.5 * erfc(-z / sqrt(2))
    cdf = 0.5 * erfc(-z / np.sqrt(2.0))
    pdf = norm.pdf(z)
    
    ei = d * cdf + sigma_safe * pdf
    
    # Where sigma is very small, EI is essentially max(d, 0)
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    return ei


def _ucb(mu, sigma, beta):
    """Upper Confidence Bound for minimization (negative of standard UCB)."""
    return mu - beta * sigma


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
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    # Compute Expected Improvement
    ei = _ei_fast(mu, sigma, f_best)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # Adaptive cost-awareness using exponential penalty instead of power law.
    # This gives smoother gradients and better calibration.
    mean_sigma = np.mean(sigma)
    # Normalize sigma: large sigma -> 1.0 (early), small sigma -> 0.0 (late)
    sigma_norm = mean_sigma / (mean_sigma + 0.05)  # in (0, 1)
    
    # Cost penalty strength: stronger when uncertainty is low (late in run)
    # Early: gamma ~ 0.3 (lenient), Late: gamma ~ 1.0 (strict)
    gamma_cost = 0.3 + 0.7 * (1.0 - sigma_norm)
    gamma_cost = np.clip(gamma_cost, 0.2, 1.2)
    
    cost_safe = np.maximum(cost, 1.0)
    # Exponential cost penalty: exp(-gamma * (cost - 1)) penalizes expensive points
    # When cost=1, penalty=1; when cost>1, penalty<1 (reduces score)
    cost_penalty = np.exp(-gamma_cost * (cost_safe - 1.0))
    
    ei_cost = ei * cost_penalty

    # UCB term for exploration, also cost-aware
    # beta adapts: larger early (explore), smaller late (exploit)
    beta_ucb = 0.5 + 0.5 * sigma_norm  # 1.0 early, 0.5 late
    ucb_term = _ucb(mu, sigma, beta_ucb)
    ucb_cost = ucb_term * cost_penalty

    # Combine EI and UCB with EI dominant
    score = 0.8 * ei_cost + 0.2 * ucb_cost
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
