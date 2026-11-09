"""Acquisition for Bayesian optimization of Ackley 2D; EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Explicit wall-clock deadline guard (evaluator kills at 330 s; keep margin).
_DEADLINE = time.time() + 325.0


def _is_finite(x):
    return np.isfinite(x).all()


def acquisition(mu, sigma, f_best):
    """Return 2000 finite acquisition scores (higher = evaluate next).

    UCB-style optimistic score: mu - kappa * sigma. The negative sign makes
    the acquisition *maximize* by preferring low predicted mean and high
    predictive uncertainty, which drives exploration in the low-budget
    Ackley landscape and helps escape local minima.
    """
    # --- Time guard: if we are too close to the deadline, fall back to a
    # deterministic pure-exploitation score so the loop always gets a valid
    # finite vector back quickly.
    if time.time() > _DEADLINE:
        return -np.asarray(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Robustify sigma: clip tiny values and drop non-finite entries.
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.clip(sigma, 0.0, None)

    # UCB exploration coefficient. A moderate value balances exploitation
    # (low mu) against exploration (high sigma) for a 30-evaluation budget.
    kappa = 2.0

    # Primary score: minimize mu - kappa*sigma  (maximize the negative).
    score = -mu + kappa * sigma

    # Safety: replace any non-finite values with a conservative fallback so
    # the returned vector is always finite.
    if not _is_finite(score):
        score = np.where(np.isfinite(score), score, -np.where(np.isfinite(mu), mu, 0.0))
        if not _is_finite(score):
            score = np.zeros_like(mu)

    return score
# EVOLVE-BLOCK-END
