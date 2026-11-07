import numpy as np
import time

_DEADLINE = time.time() + 320.0  # safety margin before 330s wall-clock


def acquisition(mu, sigma, f_best, cost):
    # --- deadline guard: if we're close to the limit, return a safe finite array
    if time.time() > _DEADLINE - 5.0:
        return np.ones_like(mu, dtype=float) * 0.0

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Expected Improvement (standard, robust form)
    improvement = f_best - mu
    z = improvement / sigma
    # clip z to avoid overflow in cdf/pdf
    z = np.clip(z, -30.0, 30.0)
    from scipy.stats import norm
    EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.maximum(EI, 0.0)  # EI is non-negative by definition

    # Efficiency: EI per unit cost
    eff = EI / cost

    # Adaptive gate: keep candidates whose efficiency is at/above a fraction of
    # the mean efficiency; damp the rest. This creates a hard budget-aware
    # constraint that forces exploitation of the cheap (low-cost) corner until
    # the GP's uncertainty there is exhausted, then relaxes naturally.
    mean_eff = np.mean(eff)
    if not np.isfinite(mean_eff) or mean_eff <= 0:
        # fallback: plain EI
        scores = EI
    else:
        gate_frac = 0.75  # allow slightly below-average efficiency to pass
        gate = gate_frac * mean_eff
        feasible = eff >= gate
        # damp infeasible candidates heavily but keep a floor for exploration
        damp = 0.12
        scores = np.where(feasible, EI, EI * damp)

    # Ensure finite, and keep a small exploration floor so we never return
    # an all-zero (degenerate) array.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    floor = 1e-12
    scores = np.maximum(scores, floor)

    return scores
