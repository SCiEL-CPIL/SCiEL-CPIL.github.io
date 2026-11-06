"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 110.0  # safety margin before 120s kill


def _check_deadline():
    return (_time.monotonic() - _DEADLINE_START) > _DEADLINE_LIMIT


def _fastest_machine(op):
    """Return (machine, duration) for the fastest option for this operation."""
    mo = op["machine_options"]
    best_m = None
    best_d = None
    for m, d in mo.items():
        if best_d is None or d < best_d:
            best_d = d
            best_m = m
    return best_m, best_d


def dispatch(ready_operations, time):
    # Deadline guard: if we're about to exceed the wall-clock limit, fall back fast.
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-compute features for each ready operation
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Fastest machine duration for each op
    fastest_durs = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        _, d = _fastest_machine(op)
        fastest_durs[i] = d

    # Avoid division by zero
    remaining_ops_safe = np.maximum(remaining_ops, 1.0)

    # Average remaining work per op (proxy for urgency/criticality)
    avg_work_per_op = remaining_work / remaining_ops_safe

    # Normalize each criterion to [0, 1] range for balanced multi-criteria weighting
    # MWR (Most Work Remaining): higher remaining_work -> higher priority
    # SPT (Shortest Processing Time): lower fastest_dur -> higher priority
    # rem_ops: higher remaining_ops -> higher priority (more work to schedule)
    # constraint: fewer machine options -> higher priority (more constrained)

    # Normalize remaining_work
    rw_min, rw_max = remaining_work.min(), remaining_work.max()
    rw_range = rw_max - rw_min
    if rw_range > 1e-9:
        mwr_norm = (remaining_work - rw_min) / rw_range
    else:
        mwr_norm = np.ones(n, dtype=np.float64) * 0.5

    # Normalize fastest_durs (invert so lower duration -> higher score)
    fd_min, fd_max = fastest_durs.min(), fastest_durs.max()
    fd_range = fd_max - fd_min
    if fd_range > 1e-9:
        spt_norm = 1.0 - (fastest_durs - fd_min) / fd_range
    else:
        spt_norm = np.ones(n, dtype=np.float64) * 0.5

    # Normalize remaining_ops
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    ro_range = ro_max - ro_min
    if ro_range > 1e-9:
        rem_ops_norm = (remaining_ops - ro_min) / ro_range
    else:
        rem_ops_norm = np.ones(n, dtype=np.float64) * 0.5

    # Constraint: fewer machine options means more constrained, so prefer fewer
    n_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    no_min, no_max = n_options.min(), n_options.max()
    no_range = no_max - no_min
    if no_range > 1e-9:
        constraint_norm = 1.0 - (n_options - no_min) / no_range
    else:
        constraint_norm = np.ones(n, dtype=np.float64) * 0.5

    # Multi-criteria weighted sum with measured effective weights
    w_mwr = 0.5
    w_spt = 0.3
    w_rem_ops = 0.15
    w_constraint = 0.05

    score = (w_mwr * mwr_norm + 
             w_spt * spt_norm + 
             w_rem_ops * rem_ops_norm + 
             w_constraint * constraint_norm)

    # Small tie-breaker: prefer jobs with more remaining work
    score += remaining_work * 1e-6

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
