import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]
    zero = np.zeros(n, dtype=np.float64)

    # Safety: if inputs are degenerate, return a neutral score.
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return zero

    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)

    # ---- Expected Improvement (minimization) ----
    imp = f_best - mu
    s = sigma_safe
    z = imp / s
    ei = imp * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # ---- Probability of Improvement ----
    poi = norm.cdf(z)

    # ---- Blend EI with PoI confidence factor ----
    # The PoI term emphasizes points where improvement is likely,
    # while EI captures the magnitude of expected improvement.
    blended = ei * (1.0 + poi)

    # ---- Cost normalization ----
    scores = blended / cost_safe

    # Ensure finite, non-negative.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)

    # Deadline guard: return the current (already computed) result safely.
    if time.time() - start_time > 325:
        return scores

    return scores
