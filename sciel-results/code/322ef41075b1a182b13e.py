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

    # ---- Adaptive exploration coefficient (shared) ----
    # Early on (high median sigma, wide spread) we explore more; late we exploit.
    med_sigma = float(np.median(sigma_safe))
    max_sigma = float(np.max(sigma_safe))
    if med_sigma > 1e-12:
        ratio = max_sigma / med_sigma
        beta = float(np.clip(0.5 * np.log1p(ratio), 0.5, 5.0))
    else:
        beta = 0.5

    # ---- Strategy 1: Expected Improvement (minimization) ----
    # Improvement = f_best - f ; EI = E[max(0, f_best - f)]
    imp = f_best - mu
    s = sigma_safe
    z = imp / s
    ei = imp * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    s1 = ei / cost_safe

    # ---- Strategy 2: LCB on improvement, cost-normalized ----
    lcb_improvement = (f_best - mu) - beta * s
    s2 = np.maximum(0.0, lcb_improvement) / cost_safe

    # ---- Strategy 3: Upper Confidence Bound on the objective (explore high sigma) ----
    # We want to minimize f, so we maximize -mu + beta*sigma, then penalize by cost.
    ucb = -mu + beta * s
    s3 = ucb / cost_safe

    # ---- Strategy 4: Pure exploitation of the GP mean (cost-adjusted) ----
    # Favor points the GP believes are genuinely low, regardless of uncertainty.
    s4 = -mu / cost_safe

    # ---- Ensemble: max across strategies ----
    # Each strategy is on a different scale; max picks the most aggressive
    # recommendation at every point, covering exploration and exploitation.
    scores = np.maximum(np.maximum(s1, s2), np.maximum(s3, s4))

    # Ensure finite, non-negative.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)

    # Deadline guard: return the current (already computed) result safely.
    if time.time() - start_time > 325:
        return scores

    return scores
