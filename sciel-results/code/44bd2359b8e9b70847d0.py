"""Cost-aware acquisition for griewank-2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Monotonic deadline guard (330 s wall-clock kill; keep a large safety margin).
_T0 = time.monotonic()
_DEADLINE = 300.0  # seconds


def _finite(x, default=0.0):
    return np.where(np.isfinite(x), x, default)


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.shape[0]

    # Safety: if we are near the deadline, return a safe finite fallback.
    if time.monotonic() - _T0 > _DEADLINE:
        return np.zeros(n, dtype=np.float64)

    # Guard against non-finite / degenerate inputs.
    mu = _finite(mu, 0.0)
    sigma = _finite(sigma, 1e-6)
    cost = _finite(cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # f_best may be a scalar or an array; normalize to a scalar-ish broadcast.
    fb = float(np.min(f_best)) if np.ndim(f_best) else float(f_best)
    if not np.isfinite(fb):
        fb = 0.0

    # --- Cost-aware Expected Improvement (clipped improvement) ---
    z = (fb - mu) / sigma
    # Numerically safe norm.cdf / norm.pdf for extreme z.
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    improve = np.maximum(fb - mu, 0.0)          # never reward already-good points
    ei = improve * cdf + sigma * pdf            # expected improvement

    # Cost-awareness: prefer cheap points (cost in [1,2] -> 1/cost in [0.5,1]).
    qei = ei / cost

    # --- Cost-aware exploration bonus (gated by relative uncertainty) ---
    # Early: sigma/fb large -> explore; late: sigma/fb small -> exploit.
    rel_unc = sigma / np.maximum(np.abs(fb), 1e-6)
    explore = (sigma / cost) * np.exp(-rel_unc)  # decays as uncertainty drops

    # Blend exploitation and exploration; keep scores finite.
    score = qei + 0.5 * explore

    # Ensure a strictly finite, real output of the right shape.
    score = np.asarray(score, dtype=np.float64)
    score = np.where(np.isfinite(score), score, 0.0)
    if score.shape != (n,):
        score = np.reshape(score, (n,))
    return score
# EVOLVE-BLOCK-END
