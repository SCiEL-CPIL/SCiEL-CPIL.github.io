"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
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
    """Fast Gaussian Expected Improvement for minimization using erfc."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    
    # norm.cdf(z) = 0.5 * erfc(-z / sqrt(2))
    cdf_z = 0.5 * erfc(-z / np.sqrt(2.0))
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = d * cdf_z + sigma_safe * pdf_z
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    return ei


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

    # Deadline guard
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    # Compute Gaussian Expected Improvement
    ei = _ei_fast(mu, sigma, f_best)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # Structural Change: Stochastic Cost-Efficiency Metric
    # Model cost as Gamma(mean=cost, shape=2). E[Cost] = cost.
    # Score = E[Improvement] / E[Cost] + Exploration Term / E[Cost]
    # This directly implements "Expected Improvement per Expected Cost".
    
    cost_safe = np.maximum(cost, 1.0)
    
    # Primary term: EI per unit cost
    ei_cost = ei / cost_safe
    
    # Exploration term: Uncertainty per unit cost
    # Scaled to be comparable to EI terms
    explore_boost = 0.1 * sigma / cost_safe
    
    score = ei_cost + explore_boost
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
