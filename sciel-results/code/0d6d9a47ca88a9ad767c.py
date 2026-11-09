import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 320.0  # seconds, safety margin before 330s kill


def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE


def _time_remaining():
    deadline = _ensure_deadline()
    return deadline - time.time()


def _sanitize(scores):
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)


def acquisition(mu, sigma, f_best):
    global _DEADLINE
    _ensure_deadline()

    # Safety
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    sigma_safe = np.maximum(sigma, 1e-6)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)

    # Pure Expected Improvement for minimization
    z = (f_best_safe - mu_safe) / sigma_safe
    ei = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    return _sanitize(ei)
