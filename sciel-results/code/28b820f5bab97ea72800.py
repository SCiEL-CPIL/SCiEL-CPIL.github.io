"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Global deadline guard
_START_TIME = time_module.perf_counter()
_DEADLINE_LIMIT = 110.0  # Safety margin before 120s limit


def _logsumexp(x, temp):
    """Temperature-scaled log-sum-exp (smooth max surrogate)."""
    x = np.asarray(x, dtype=float)
    m = x.max(axis=1, keepdims=True)
    # guard against empty rows (shouldn't happen, but be safe)
    m = np.where(np.isfinite(m), m, 0.0)
    return m.squeeze(1) + np.log(np.sum(np.exp((x - m) / temp), axis=1)) * temp


def _softmin(x, temp):
    """Temperature-scaled soft-min (smooth min surrogate)."""
    return -_logsumexp(-x, temp)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard
    if time_module.perf_counter() - _START_TIME > _DEADLINE_LIMIT:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # Urgency: average remaining work per remaining operation (measured improvement)
    # Avoid division by zero.
    ops_safe = np.where(remaining_ops > 0, remaining_ops, 1.0)
    urgency = remaining_work / ops_safe

    # Collect machine durations per operation
    n_machines = max(len(op["machine_options"]) for op in ready_operations)
    durs = np.zeros((n, n_machines), dtype=float)
    counts = np.zeros(n, dtype=int)
    for i, op in enumerate(ready_operations):
        vals = list(op["machine_options"].values())
        c = len(vals)
        counts[i] = c
        if c > 0:
            durs[i, :c] = vals

    # Smooth max (log-sum-exp) and soft-min surrogates
    TEMP = 1.0
    smooth_max = _logsumexp(durs, TEMP)
    soft_min = _softmin(durs, TEMP)

    # Hard min/max for tie-breaking flexibility (spread)
    hard_min = np.where(counts > 0, np.min(durs, axis=1), 0.0)
    hard_max = np.where(counts > 0, np.max(durs, axis=1), 0.0)
    spread = np.where(counts > 0, hard_max - hard_min, 0.0)

    # Min-max normalize each feature to [0, 1]
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    urg_norm = normalize(urgency)
    rw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    sm_norm = normalize(soft_min)
    sx_norm = normalize(smooth_max)
    spread_norm = normalize(spread)

    # Weighted score: prioritize urgency (avg work/op), then remaining work,
    # then remaining ops, then LONGER soft-min (LPT-style inflexibility signal),
    # with a REDUCED penalty on long smooth-max (flexibility).
    w_urg = 0.35
    w_work = 0.20
    w_ops = 0.15
    w_min = 0.20
    w_max = 0.10  # reduced flexibility penalty

    score = (w_urg * urg_norm +
             w_work * rw_norm +
             w_ops * ro_norm +
             w_min * sm_norm +
             w_max * (1.0 - sx_norm))

    # Tie-break: among (near-)tied scores, prefer LOWER spread (more flexible
    # ops) to reduce machine conflicts. Use a tiny perturbation.
    eps = 1e-9
    best = int(np.argmax(score))
    best_score = score[best]
    tie_mask = np.abs(score - best_score) < 1e-6
    if tie_mask.sum() > 1:
        candidates = np.where(tie_mask)[0]
        # pick candidate with lowest spread (most flexible)
        best = int(candidates[np.argmin(spread[candidates])])

    return best
# EVOLVE-BLOCK-END
