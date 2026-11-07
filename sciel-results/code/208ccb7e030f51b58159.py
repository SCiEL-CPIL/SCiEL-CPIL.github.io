"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s wall-clock limit


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)

    # Cost is in [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)

    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to prevent overflow in cdf/pdf
    z = np.clip(z, -50, 50)

    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is norm.cdf, phi is norm.pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # EI is non-negative by definition, but numerical errors might make it slightly negative
    ei = np.maximum(ei, 0.0)

    # Cost-aware score: Maximize EI per unit cost
    # This prioritizes points that offer high improvement for low cost
    score = ei / cost

    # Add a small exploration bonus based on uncertainty to avoid getting stuck
    # in exploitation-only mode, especially early in the search
    # The bonus is scaled by the relative uncertainty compared to the current best
    uncertainty_bonus = 0.1 * sigma / (np.abs(f_best) + 1e-6)
    score = score + uncertainty_bonus / cost

    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
