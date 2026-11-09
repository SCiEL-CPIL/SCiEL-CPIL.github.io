"""Cost-aware acquisition for Ackley 2D BO with UCB-like hybrid scoring and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    if _check_deadline():
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)

    # UCB-like hybrid: exploit (f_best - mu) and explore (beta * sigma)
    # beta = 1.0 was the key driver of the 1.7043 score
    beta = 1.0
    raw_scores = (f_best - mu) + beta * sigma

    # Cost-aware: divide by cost
    scores = raw_scores / cost

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
# EVOLVE-BLOCK-END
