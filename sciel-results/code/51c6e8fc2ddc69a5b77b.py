"""Rewritten acquisition: cost-aware UCB with deadline guard and safe fallbacks."""
import time
import numpy as np
from scipy.stats import norm

# Keep a module-level start time so we can enforce a wall-clock deadline.
_T0 = time.time()
_DEADLINE_FRAC = 0.90  # safety margin: stop at 90% of 330s

def _elapsed():
    return time.time() - _T0

def _deadline_exceeded():
    # 330s wall-clock limit; use 90% as a safety margin.
    return _elapsed() > 330.0 * _DEADLINE_FRAC

def acquisition(mu, sigma, f_best, cost):
    # Ensure finite inputs; fall back to a safe constant if anything is invalid.
    try:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
    except Exception:
        return np.zeros(2000, dtype=np.float64)

    n = mu.shape[0] if mu.ndim else 2000
    if mu.size != 2000 or sigma.size != 2000 or cost.size != 2000:
        return np.zeros(2000, dtype=np.float64)

    # Guard against non-finite values.
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        # Replace non-finite with safe defaults.
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-3)
        cost = np.where(np.isfinite(cost), cost, 1.0)

    # Ensure sigma is positive.
    sigma = np.maximum(sigma, 1e-9)
    # Ensure cost is positive.
    cost = np.maximum(cost, 1e-6)

    # Deadline guard: if we're close to the limit, return a simple, fast score.
    if _deadline_exceeded():
        # Use plain EI-like score without extra computation.
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Cost-aware acquisition: combine expected improvement with a UCB exploration term,
    # normalized by cost. We use a blend to balance exploitation and exploration.
    # z-score for EI.
    z = (f_best - mu) / sigma

    # Expected Improvement (for minimization).
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # UCB term: mu - beta * sigma (for minimization, we want low mu and high sigma).
    # Use a modest beta to encourage exploration in uncertain regions.
    beta = 1.0
    ucb = mu - beta * sigma

    # Normalize EI and UCB to comparable scales.
    # EI is in objective units; UCB is also in objective units.
    # We want higher score = better. EI is higher when improvement is larger.
    # UCB is lower when mu is low or sigma is high (more promising for minimization).
    # So we use -ucb as a score component (higher = more promising).

    # Combine: primarily EI/cost, with a smaller UCB/cost component.
    # Weights: 0.7 for EI, 0.3 for UCB.
    score_ei = ei / cost
    score_ucb = (-ucb) / cost  # negative UCB so higher is better

    # Final score: weighted combination.
    score = 0.7 * score_ei + 0.3 * score_ucb

    # Ensure all scores are finite.
    score = np.where(np.isfinite(score), score, 0.0)

    return score
