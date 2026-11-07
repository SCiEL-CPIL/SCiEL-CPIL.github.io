"""Cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 300.0          # evaluator wall-clock limit
_SAFETY   = 15.0           # return well before the limit
_start    = time.time()


def _elapsed():
    return time.time() - _start


def _ei_scores(mu, sigma, f_best, cost, xi):
    """Expected improvement (minimization) with exploration bonus xi."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    s = np.where(sigma > 1e-12, sigma, 1e-12)
    gain = f_best - xi - mu
    z = gain / s
    # EI = gain * Phi(z) + s * phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = gain * cdf + s * pdf
    # guard against tiny negative from round-off
    ei = np.where(ei > 0.0, ei, 0.0)
    # cost-aware: EI per unit cost
    c = np.where(cost > 1e-12, cost, 1e-12)
    return ei / c


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = mu.shape[0]

    # Ensure finiteness of inputs we depend on.
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all() and np.isfinite(cost).all()):
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)
        cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
        sigma = np.clip(sigma, 1e-9, None)
        cost = np.clip(cost, 1e-9, None)

    # Base score: cost-aware EI with a mild exploration bonus.
    base = _ei_scores(mu, sigma, f_best, cost, xi=0.0)

    # Exploration bonus: reward high uncertainty (sigma) so we keep exploring
    # where the GP is unsure, scaled by 1/cost. This is a structural addition
    # over pure cost-normalized EI and helps escape local minima.
    s = np.where(sigma > 1e-12, sigma, 1e-12)
    c = np.where(cost > 1e-12, cost, 1e-12)
    explore = (s / c)

    # Blend: EI dominates, exploration is a secondary signal.
    # Normalize explore to the EI scale so it does not dominate.
    ei_max = np.max(base) if base.size else 0.0
    ex_max = np.max(explore) if explore.size else 0.0
    if ex_max > 0 and ei_max > 0:
        explore = explore * (ei_max / ex_max) * 0.5
    scores = base + explore

    # Final safety: guarantee finiteness.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.clip(scores, -1e12, 1e12)

    # Deadline guard: if we are dangerously close to the limit, still return
    # a valid finite vector (we already have one). No extra work needed.
    if _elapsed() > _DEADLINE - _SAFETY:
        return scores

    return scores
# EVOLVE-BLOCK-END
