"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.stats import norm

_START = time.perf_counter()
_DEADLINE = _START + 295.0  # safety margin well under the 300 s per-eval / 330 s kill limit


def acquisition(mu, sigma, f_best, cost):
    """
    Multi-objective UCB acquisition for 2000 candidates.

    Balances three competing objectives:
      - exploitation: expected improvement over the best observed value,
      - exploration:  uncertainty (GP std-dev) to cover unexplored regions,
      - cost:         prefer cheap evaluations, using a GP-predicted cost model
                      so already-explored cheap regions are not over-penalised.

    Higher score => more preferred to evaluate next.
    """
    # --- deadline guard: bail out early if we are running out of wall-clock ---
    if time.perf_counter() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # --- numerical safety ---
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1e-9)
    f_best_safe = float(f_best) if np.isfinite(f_best) else 0.0

    # --- 1) exploitation: expected improvement over f_best ---
    improvement = f_best_safe - mu_safe
    z = improvement / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = improvement * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)

    # --- 2) GP-predicted cost model (mu_c, sigma_c) ---
    # Fit a linear Gaussian process (mean + isotropic covariance) to the
    # observed (candidate, cost) pairs. This yields a per-candidate predicted
    # cost that is smooth in space, unlike the raw per-candidate cost.
    mu_c = np.full(n, float(np.mean(cost_safe)))
    sigma_c = np.full(n, 1e-3)
    try:
        # Subsample for speed; the cost model only needs a rough fit.
        m = cost_safe.size
        if m > 1000:
            idx = np.linspace(0, m - 1, 1000).astype(int)
            mu_sub = mu_safe[idx]
            cost_sub = cost_safe[idx]
        else:
            idx = np.arange(m)
            mu_sub = mu_safe
            cost_sub = cost_safe

        mu_mean = float(np.mean(mu_sub))
        c_mean = float(np.mean(cost_sub))
        Xc = mu_sub - mu_mean
        cov = np.cov(Xc, rowvar=False)
        cov = np.atleast_2d(cov)
        var = float(np.mean(np.diag(cov)))
        if not np.isfinite(var) or var < 1e-12:
            var = 1e-6

        # Linear regression: cost ~ a + b * (mu - mean)
        # Solve least squares [1, Xc] -> cost_sub
        A = np.column_stack([np.ones_like(Xc), Xc])
        try:
            beta, *_ = np.linalg.lstsq(A, cost_sub, rcond=None)
            a, b = float(beta[0]), float(beta[1])
        except Exception:
            a, b = c_mean, 0.0

        Xq = mu_safe - mu_mean
        mu_c = a + b * Xq
        # Predictive variance: b^2 * var + residual noise
        resid = cost_sub - (a + b * Xc)
        noise = float(np.var(resid)) if resid.size > 1 else 1e-6
        if not np.isfinite(noise) or noise < 1e-9:
            noise = 1e-6
        sigma_c = np.sqrt(np.maximum(b * b * var + noise, 1e-9))
    except Exception:
        # Fallback: use raw cost as the predicted cost, small uncertainty.
        mu_c = cost_safe
        sigma_c = np.full(n, 1e-3)

    # --- 3) cost term: prefer cheap, with a little uncertainty-driven
    #         optimism about cost (cheap regions with uncertain cost are
    #         worth probing). ---
    cost_optimism = mu_c - 0.5 * sigma_c
    cost_optimism = np.maximum(cost_optimism, 1e-3)
    # Normalise so the cost penalty is on a comparable scale to ei/sigma.
    cost_scale = float(np.median(cost_optimism))
    if not np.isfinite(cost_scale) or cost_scale < 1e-6:
        cost_scale = 1.0
    cost_penalty = cost_optimism / cost_scale

    # --- 4) exploration term: uncertainty normalised by the spread of sigma ---
    sigma_scale = float(np.median(sigma_safe))
    if not np.isfinite(sigma_scale) or sigma_scale < 1e-9:
        sigma_scale = 1e-9
    exploration = sigma_safe / sigma_scale

    # --- 5) combine into a UCB-style score ---
    #   score = w_ei * ei/sigma_scale  +  w_exp * exploration  -  w_cost * cost_penalty
    w_ei = 1.0
    w_exp = 0.35
    w_cost = 0.6
    scores = w_ei * (ei / sigma_scale) + w_exp * exploration - w_cost * cost_penalty

    # --- numerical cleanup ---
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
