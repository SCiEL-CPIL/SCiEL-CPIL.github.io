import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 320.0  # safety margin before 330s kill


def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Handle shape robustness
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)
    else:
        P = prizes

    if weights.ndim == 2:
        W = weights
    else:
        W = weights.reshape(1, -1)

    n_constraints, n_items = W.shape
    eps = 1e-9

    # Per-constraint efficiency: prize / weight
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = P[None, :] / (W + eps)
    eff = np.where(np.isfinite(eff), eff, 0.0)

    # --- Scarcity weighting (current program's core) ---
    # Constraints with higher top-k efficiency are "scarcer" (best items use
    # them most efficiently, so they bind first).
    k = max(1, n_items // 5)  # top 20%
    top_k_eff = np.zeros(n_constraints)
    for c in range(n_constraints):
        top_k_eff[c] = np.partition(eff[c], -k)[-k].mean()

    # Also factor in total weight scarcity (tighter constraint = less slack)
    weight_sum = W.sum(axis=1)  # (5,)
    # Combine: higher top-k eff AND lower total weight => more scarce
    raw_scarcity = top_k_eff * (1.0 / (weight_sum + eps))
    raw_scarcity = np.where(np.isfinite(raw_scarcity), raw_scarcity, 0.0)
    scarcity = raw_scarcity / (raw_scarcity.sum() + eps)  # normalized, sums to 1

    # --- Smooth surrogate blend (the refinement machinery) ---
    # Instead of a hard product, use a soft-min / log-sum-exp style surrogate.
    #
    # Metric A: scarcity-weighted marginal utility (linear, current design)
    marginal_util = (scarcity[:, None] * eff).sum(axis=0)  # (100,)

    # Metric B: bottleneck efficiency via smooth geometric mean in log-space.
    # Weighted by scarcity so the tightest constraints dominate the "bottleneck".
    log_eff = np.log(eff + eps)  # (5, 100)
    # Smooth weighted geometric mean: exp(sum_c scarcity_c * log(eff_c))
    weighted_log = (scarcity[:, None] * log_eff).sum(axis=0)  # (100,)
    geo_ratio = np.exp(weighted_log)  # (100,)

    # Base efficiency (prize per total weight) as a stabilizing scale
    W_sum = W.sum(axis=0)
    base_eff = P / (W_sum + eps)
    base_eff = np.where(np.isfinite(base_eff), base_eff, 0.0)

    # Normalize each metric to mean 1 so they blend fairly (prevents scale
    # dominance by either the linear or the geometric term).
    def _normalize(x):
        m = x.mean()
        if m > 0 and np.isfinite(m):
            return x / m
        return np.ones_like(x)

    mu_norm = _normalize(marginal_util)
    geo_norm = _normalize(geo_ratio)

    # Smooth blend via a soft combination: geometric mean of the two normalized
    # metrics is equivalent to a log-space average, which is a smooth surrogate
    # for "both must be good" (soft-min-like behavior). We use a weighted
    # log-average with a tunable balance, then re-scale by base efficiency to
    # keep magnitude comparable to the original design.
    alpha = 0.5  # balance between linear marginal utility and bottleneck geo
    log_blend = alpha * np.log(mu_norm + eps) + (1.0 - alpha) * np.log(geo_norm + eps)
    blend = np.exp(log_blend)  # smooth geometric blend

    # Final heuristic: smooth blend scaled by base efficiency
    H = blend * base_eff

    # Ensure positive and finite
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)

    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        return H

    return H
