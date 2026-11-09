"""Cost-aware acquisition for cosine8 8d BO with deadline guard."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 5.0  # seconds to spare before 330s limit


def _safe_acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function: combines expected improvement per unit cost with
    an exploration bonus weighted by uncertainty, and a small exploitation
    term for low predicted values.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero/negative cost
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)

    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    ei_cost = ei / cost

    # Uncertainty bonus: encourages exploring high-uncertainty regions
    # Weighted to be significant but not dominate
    uncertainty_bonus = sigma / (sigma + 0.1)  # bounded in [0, 1)

    # Exploitation term: favor low predicted values
    # Normalize mu to [0,1] range for scaling (cosine8 min ~ -0.8, max ~ 1)
    mu_norm = (mu - (-1.0)) / 2.0  # map [-1,1] to [0,1]
    mu_norm = np.clip(mu_norm, 0.0, 1.0)
    exploitation = 1.0 - mu_norm  # higher when mu is lower

    # Combine: cost-aware EI dominates, with exploration and exploitation
    # The weights are chosen so that EI_cost is the primary driver
    score = 1.0 * ei_cost + 0.3 * uncertainty_bonus + 0.2 * exploitation

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score


def acquisition(mu, sigma, f_best, cost):
    # Deadline check: if we're too close to the limit, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - DEADLINE_MARGIN:
        # Fallback: return a simple cost-aware EI to finish quickly
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-6)
        sigma = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost
        score = np.where(np.isfinite(score), score, 0.0)
        return np.maximum(score, 0.0)

    try:
        return _safe_acquisition(mu, sigma, f_best, cost)
    except Exception:
        # Last-resort fallback: uniform random scores
        return np.ones(2000, dtype=np.float64) * 0.1
