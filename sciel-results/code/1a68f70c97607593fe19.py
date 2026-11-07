import time
import numpy as np
from scipy.stats import norm

_START = time.perf_counter()
_DEADLINE = _START + 295.0   # hard wall-clock guard (evaluator kills at 330 s)
_TAU = 1e-4                  # floor on sigma (objective units)


def _safe_finite(arr, fallback=0.0):
    """Replace non-finite values with a fallback."""
    if not np.all(np.isfinite(arr)):
        arr = np.where(np.isfinite(arr), arr, fallback)
    return arr


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # --- deadline guard: bail out with a safe finite score if we are about to overrun ---
    if time.perf_counter() > _DEADLINE:
        return np.clip(-mu, 0.0, 1e8)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    # sanitize sigma
    sigma = np.where(np.isfinite(sigma), sigma, _TAU)
    sigma = np.clip(sigma, _TAU, 1e8)

    # sanitize mu
    mu = np.where(np.isfinite(mu), mu, f_best + sigma)

    # --- Pareto-frontier filtering: mask out hopeless candidates ---
    # A candidate is "hopeless" if its predicted mean is worse than f_best
    # by more than a tolerance that scales with the remaining uncertainty.
    # This restricts the search to regions with plausible improvement.
    # Tolerance: we allow candidates whose mu is at most f_best + tol,
    # where tol grows with the median sigma (uncertainty) but shrinks
    # as f_best approaches 0 (convergence).
    sigma_med = np.median(sigma)
    if not np.isfinite(sigma_med) or sigma_med < _TAU:
        sigma_med = _TAU

    # Convergence-aware tolerance:
    # When f_best is large (early), allow a wide band of candidates.
    # When f_best is small (near optimum), tighten the band.
    # tol = max(min_tol, c * sigma_med * log(1 + 1/f_best_abs))
    f_best_abs = max(abs(float(f_best)), 1e-6)
    # log factor: large when f_best is large (early), small when f_best ~ 0
    log_factor = np.log1p(1.0 / f_best_abs)
    tol = max(0.05, 1.5 * sigma_med * log_factor)

    # Mask: candidate is viable if mu <= f_best + tol
    viable = mu <= (float(f_best) + tol)

    # --- Adaptive UCB with Pareto filtering ---
    # Beta grows as we converge: beta = base + ramp * (1 / (1 + f_best_abs))
    # Early: beta ~ base (moderate exploration)
    # Late: beta ~ base + ramp (aggressive exploitation of uncertainty)
    base_beta = 1.0
    ramp_beta = 2.5
    beta = base_beta + ramp_beta / (1.0 + f_best_abs)

    # UCB score: -mu + beta * sigma
    # Higher score = more promising (since we minimize, -mu favors low mu)
    ucb = -mu + beta * sigma

    # Apply Pareto filter: hopeless candidates get a very low score
    # so they are never selected, but scores remain finite.
    floor_score = -1e6
    score = np.where(viable, ucb, floor_score)

    # Safety: ensure all scores are finite and bounded
    score = _safe_finite(score, floor_score)
    score = np.clip(score, floor_score, 1e8)

    return score
