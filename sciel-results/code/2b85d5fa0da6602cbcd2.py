"""Cost-aware acquisition function for Bayesian optimization (ackley 2d).

Thompson-Sampling variant: sample a single scalar threshold T from a Gaussian
centred at f_best whose spread scales with the candidate-set uncertainty, then
score every candidate as (T - mu) / cost.  Maximising this stochastically
balances exploration and exploitation while rewarding low-cost evaluations.
"""
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against pathological inputs while keeping shapes (2000,).
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)

    # Reference scale: use the bulk of the uncertainty (robust to a few
    # outlier candidates) rather than the raw maximum, which can over-exploration.
    spread = np.percentile(sigma, 90)
    spread = max(spread, 1e-12)

    # Sample a single threshold.  A coefficient of 0.8 gives a moderately
    # aggressive (exploitation-leaning) draw while still allowing the
    # threshold to fall below f_best and trigger exploration.
    T = f_best + 0.8 * spread * np.random.standard_normal()

    # Score: prefer candidates whose predicted mean is well below the
    # sampled threshold, divided by evaluation cost.  Higher = evaluate next.
    scores = (T - mu) / cost

    # Enforce finiteness (anything non-finite invalidates the result).
    scores = np.where(np.isfinite(scores), scores, -1e30)

    return scores
