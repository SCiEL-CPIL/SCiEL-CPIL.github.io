# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: adaptive UCB with an exploitation floor.
    mu, sigma : (2000,) arrays, sigma is the GP posterior std (objective units)
    f_best    : lowest observed objective value
    Returns 2000 finite scores; higher => evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-9)

    # Progress estimate: early in optimization the GP is uncertain (high mean sigma),
    # later it is confident (low mean sigma). Map to a decay factor in [0, 1].
    mean_sigma = float(np.mean(sigma))
    # Typical sigma at start for standardized GP ~ O(1); near the end ~ 1e-3..1e-2.
    # Use a logistic-like decay so beta goes from ~3.0 down to ~0.5.
    # Scale: when mean_sigma ~ 0.5 (early) -> t~0.9; when mean_sigma ~ 0.02 (late) -> t~0.05
    t = np.clip(1.0 - (mean_sigma / 0.6), 0.0, 1.0)
    # beta: high exploration early, low late
    beta = 0.5 + 2.5 * (1.0 - t)

    # UCB: maximize (f_best - mu) + beta * sigma  (since lower objective is better,
    # we want to minimize mu and explore where sigma is high).
    # Use a robust centering: exploit the best mean, add exploration.
    ucb = (f_best - mu) + beta * sigma

    # Add a small exploitation push toward the current best mean to break ties
    # and avoid purely random selection when sigma is uniform.
    mu_min = float(np.min(mu))
    exploit = (mu_min - mu) * 0.1  # small, keeps ordering stable

    scores = ucb + exploit

    # Ensure finiteness and a finite spread (avoid all-equal scores).
    scores = np.where(np.isfinite(scores), scores, 0.0)
    if np.allclose(scores, scores[0]):
        # Fallback: rank by sigma (explore) with tiny noise-free tiebreak
        scores = sigma + 1e-12 * np.arange(len(sigma))

    return scores
# EVOLVE-BLOCK-END
