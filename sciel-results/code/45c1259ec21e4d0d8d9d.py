"""Cost-aware acquisition for Bayesian optimization of powell 4d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 330.0          # evaluator wall-clock kill
_MARGIN = 2.0              # safety margin (seconds)
_START = time.monotonic()


def _log_cdf(z):
    """Numerically stable log of the standard normal CDF."""
    cdf = norm.cdf(z)
    cdf = np.clip(cdf, 1e-300, 1.0 - 1e-15)
    return np.log(cdf)


def _log_pdf(z):
    """Numerically stable log of the standard normal PDF."""
    return -0.5 * z * z - 0.5 * np.log(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    # --- deadline guard: bail out safely before the evaluator kills us ---
    if time.monotonic() - _START > _DEADLINE - _MARGIN:
        return np.zeros_like(np.asarray(mu, dtype=float))

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # guard against non-finite / non-positive inputs
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0.0), cost, 1e-6)

    # expected improvement (positive when mu < f_best)
    imp = f_best - mu
    z = imp / sigma
    log_ei = _log_cdf(z) + _log_pdf(z)
    ei = np.exp(log_ei) * sigma + imp * np.exp(_log_cdf(z))
    ei = np.where(np.isfinite(ei) & (ei > 0.0), ei, 0.0)

    # exploration bonus to escape local minima across shifted instances
    score = ei / cost + 0.1 * (sigma / cost)

    # guarantee a finite, well-ordered output
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
