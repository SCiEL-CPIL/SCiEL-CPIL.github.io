# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).
    Uses Probability of Improvement (PoI) weighted by inverse cost, plus a
    small uncertainty bonus. This prioritizes candidates with high likelihood
    of improvement per unit cost rather than expected magnitude of improvement.
        score(x) = PoI(x) / cost(x) + gamma * sigma(x) / cost(x)
    """
    t_start = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.size

    # Numerical guards
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Robust problem scale from the surrogate (dimensionless conditioning)
    finite_mu = mu[np.isfinite(mu)]
    if finite_mu.size >= 2:
        scale = float(np.std(finite_mu))
    else:
        scale = float(np.max(np.abs(finite_mu))) if finite_mu.size else 0.0
    if not np.isfinite(scale) or scale < 1e-6:
        abs_best = float(np.abs(f_best)) if np.isfinite(f_best) else 0.0
        scale = max(abs_best, 1e-6)
    scale = float(np.clip(scale, 1e-6, 1e6))

    # Probability of Improvement for minimization: P(y < f_best), y ~ N(mu, sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -40.0, 40.0)
    poi = norm.cdf(z)
    poi = np.maximum(poi, 0.0)
    poi = np.where(np.isfinite(poi), poi, 0.0)

    # Cost-aware probability of improvement (likelihood per unit cost)
    poi_cost = poi / cost_safe

    # Auto-calibrate exploration weight to the typical magnitude of poi_cost
    finite_pc = poi_cost[np.isfinite(poi_cost) & (poi_cost > 0)]
    if finite_pc.size:
        typical = float(np.median(finite_pc))
    else:
        typical = 0.0
    if not np.isfinite(typical) or typical <= 1e-12:
        typical = max(scale / 1e8, 1e-8)

    typical_std_cost = float(np.median(sigma_safe / cost_safe))
    if not np.isfinite(typical_std_cost) or typical_std_cost <= 0:
        typical_std_cost = 1.0
    gamma = np.clip(0.1 * typical / typical_std_cost, 0.0, 0.5)

    exploration = gamma * (sigma_safe / cost_safe)

    scores = poi_cost + exploration

    # Ensure all values are finite (higher = evaluate next)
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard (safety margin before the 330 s kill)
    if (time.time() - t_start) > 300.0:
        return scores

    return scores
# EVOLVE-BLOCK-END
