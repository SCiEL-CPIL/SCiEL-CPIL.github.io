"""Job-Shop Scheduling Dispatch Rule.
Refinement: Ratio-based priority with normalized secondary features.
Primary: min_duration / (1 + remaining_work) — prefer short ops on critical jobs.
Secondary: normalized avg remaining duration (critical path proxy) and flexibility.
Combined with tuned weights for better discrimination.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 110.0  # Safety margin against 150s limit


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # Deadline guard: fall back to pure SPT with best machine
    if _time.time() - _START_TS > DEADLINE_S:
        min_durs = np.empty(n, dtype=np.float64)
        best_m = [None] * n
        for i, op in enumerate(ready_operations):
            mo = op.get("machine_options", {})
            if mo:
                bm, bd = min(mo.items(), key=lambda kv: kv[1])
                min_durs[i] = bd
                best_m[i] = bm
            else:
                min_durs[i] = float("inf")
        idx = int(np.argmin(min_durs))
        if best_m[idx] is not None:
            return (idx, best_m[idx])
        return idx

    # ---- Extract features ----
    min_durs = np.empty(n, dtype=np.float64)
    best_mach = [None] * n
    rws = np.empty(n, dtype=np.float64)
    ros = np.empty(n, dtype=np.float64)
    n_opts = np.empty(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            bm, bd = min(mo.items(), key=lambda kv: kv[1])
            min_durs[i] = float(bd)
            best_mach[i] = bm
            n_opts[i] = len(mo)
        else:
            min_durs[i] = float("inf")
            best_mach[i] = None
            n_opts[i] = 0.0

        rws[i] = float(op.get("remaining_work", 0.0))
        ros[i] = float(max(op.get("remaining_ops", 1), 1))

    # ---- Compute normalized features ----
    # Primary: ratio score (lower is better)
    # score_primary = min_dur / (1 + rw)
    primary = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isinf(min_durs[i]):
            primary[i] = float("inf")
        else:
            primary[i] = min_durs[i] / (1.0 + rws[i])

    # Normalize primary to [0, 1] range (for weighting)
    p_finite = primary[np.isfinite(primary)]
    if len(p_finite) > 0:
        p_min = np.min(p_finite)
        p_max = np.max(p_finite)
        p_range = p_max - p_min
        if p_range > 1e-12:
            primary_norm = (primary - p_min) / p_range
        else:
            primary_norm = np.zeros(n, dtype=np.float64)
        primary_norm[~np.isfinite(primary)] = 1.0  # worst
    else:
        primary_norm = np.zeros(n, dtype=np.float64)

    # Secondary 1: Critical path proxy = avg remaining duration = rw / ro
    avg_durs = rws / ros
    a_max = np.max(avg_durs) if np.max(avg_durs) > 0 else 1.0
    # Higher avg_dur means more work left per op → more critical → we want to prioritize
    # So we use 1 - normalized to make "more critical" = lower score
    crit_proxy = 1.0 - (avg_durs / a_max)

    # Secondary 2: Flexibility = number of machine options
    no_max = np.max(n_opts) if np.max(n_opts) > 0 else 1.0
    flexibility = n_opts / no_max
    # More flexibility is slightly better (less bottleneck risk)
    # Use 1 - normalized to make "more flexible" = lower score
    flex_score = 1.0 - flexibility

    # ---- Combine with weights ----
    # Primary ratio is the dominant signal
    # Critical path and flexibility are secondary
    w_primary = 0.75
    w_crit = 0.15
    w_flex = 0.10

    scores = w_primary * primary_norm + w_crit * crit_proxy + w_flex * flex_score

    # Select operation with minimum score
    best_idx = int(np.argmin(scores))

    bm = best_mach[best_idx]
    if bm is not None:
        return (best_idx, bm)
    return best_idx

# EVOLVE-BLOCK-END
