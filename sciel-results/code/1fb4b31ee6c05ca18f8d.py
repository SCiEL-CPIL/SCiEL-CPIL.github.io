"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE_START = None
_DEADLINE_LIMIT = 115.0  # safety margin before 120s kill


def _ensure_deadline():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = _time.monotonic()
    if _time.monotonic() - _DEADLINE_START > _DEADLINE_LIMIT:
        return False
    return True


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0

    # Deadline guard
    if not _ensure_deadline():
        return 0

    # Precompute per-operation features
    rem_works = np.zeros(n, dtype=np.float64)
    rem_ops = np.zeros(n, dtype=np.float64)
    min_durs = np.zeros(n, dtype=np.float64)
    max_durs = np.zeros(n, dtype=np.float64)
    avg_durs = np.zeros(n, dtype=np.float64)
    num_machines = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        rem_works[i] = float(op["remaining_work"])
        rem_ops[i] = max(float(op["remaining_ops"]), 1.0)
        machine_opts = op["machine_options"]
        if machine_opts:
            vals = np.array(list(machine_opts.values()), dtype=np.float64)
            min_durs[i] = float(np.min(vals))
            max_durs[i] = float(np.max(vals))
            avg_durs[i] = float(np.mean(vals))
            num_machines[i] = float(len(machine_opts))
        else:
            min_durs[i] = 0.0
            max_durs[i] = 0.0
            avg_durs[i] = 0.0
            num_machines[i] = 1.0

    # Normalize features to [0, 1] for consistent weighting
    def normalize(arr):
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    nw_rem_work = normalize(rem_works)
    nw_rem_ops = normalize(rem_ops)
    nw_min_dur = normalize(min_durs)
    nw_max_dur = normalize(max_durs)
    nw_avg_dur = normalize(avg_durs)
    nw_num_mach = normalize(num_machines)

    # Composite score:
    # 1. Remaining work (urgency) - higher is better
    # 2. Max duration (LPT - long ops first) - higher is better
    # 3. Number of machine options (flexibility) - higher is better
    # 4. Remaining ops (fewer remaining = closer to completion) - lower is better
    # 5. Min duration (shortest available) - lower is better for quick start

    score = (
        1.0 * nw_rem_work +      # urgency: more remaining work = more critical
        0.9 * nw_max_dur +       # LPT: longer operations should start early
        0.3 * nw_num_mach +      # flexibility: more machine options = easier to schedule
        - 0.4 * nw_rem_ops +     # completion: fewer remaining ops = closer to done
        - 0.2 * nw_min_dur       # quick start: shorter min duration = less blocking
    )

    # Tie-breaking: prefer operations from jobs with fewer remaining ops
    # This helps complete jobs and free up resources
    for i in range(n):
        score[i] -= 0.05 * rem_ops[i] / max(float(np.max(rem_ops)), 1.0)

    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END
