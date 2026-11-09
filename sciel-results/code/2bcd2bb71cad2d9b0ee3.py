# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# ---- module-level deadline guard ----
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # safety margin before 330s kill


def _check_deadline():
    """Return True if we are close to the deadline."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Cost-Adjusted Probability of
    Improvement (PoI) with a GP uncertainty bonus, normalized by evaluation cost.

    Parameters
    ----------
    mu     : (2000,) GP mean predictions (objective units)
    sigma  : (2000,) GP standard deviation (objective units)
    f_best : float  best (lowest) observed value
    cost   : (2000,) cost of evaluating each candidate

    Returns
    -------
    scores : (2000,) finite, higher = evaluate next
    """
    # Deadline guard: if we're running out of time, return a fast fallback
    if _check_deadline():
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-10)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-10)
        mu = np.asarray(mu, dtype=np.float64)
        return (f_best - mu + 1.5 * sigma_safe) / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.size
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    f_best = float(f_best)

    # ---- Probability of Improvement (PoI) for minimization ----
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    delta = f_best - mu
    z = delta / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    poi = norm.cdf(z_clipped)

    # ---- Cost-Adjusted PoI: prioritize high-likelihood-of-success per unit cost ----
    cmean = float(np.mean(cost_safe))
    if cmean <= 0.0:
        cmean = 1.0
    # Normalize cost so that mean cost = 1; cheaper points get boosted
    cost_norm = cost_safe / cmean
    poi_adjusted = poi / cost_norm

    # ---- GP uncertainty bonus: encourage exploration in high-uncertainty regions ----
    # Scale by 1/cost so that cheap, uncertain points are preferred
    beta_explore = 0.5
    expl_bonus = beta_explore * sigma_safe / cost_norm

    # ---- Combine: cost-adjusted PoI + exploration bonus ----
    scores = poi_adjusted + expl_bonus

    # ---- Guard finiteness ----
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e9)

    return scores
# EVOLVE-BLOCK-END
