"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0


def _check_deadline():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_works = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    job_ids = np.empty(n, dtype=np.int64)

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        durations = list(op["machine_options"].values())
        remaining_works[i] = rw
        remaining_ops[i] = ro
        min_durations[i] = min(durations) if durations else 0.0
        job_ids[i] = op["job"]

    # --- Non-linear composite: urgency * (1 / blocking_penalty) ---
    # Urgency: jobs with more remaining work are more critical to the makespan.
    # We use a power transform to amplify differences at the high end.
    rw_max = remaining_works.max()
    if rw_max > 0:
        urgency = (remaining_works / rw_max) ** 2.0
    else:
        urgency = np.ones(n, dtype=np.float64)

    # Blocking penalty: longer operations block machines longer.
    # We want to penalize long operations, so we use 1/(1 + scaled_duration)
    # This creates a non-linear preference for short ops that saturates.
    md_max = min_durations.max()
    if md_max > 0:
        # Scale so that the longest op gets a penalty of ~0.5, shortest ~1.0
        scaled_d = min_durations / md_max
        blocking = 1.0 / (1.0 + scaled_d)
    else:
        blocking = np.ones(n, dtype=np.float64)

    # Secondary tie-breaker: fewer remaining ops (closer to completion)
    ro_max = remaining_ops.max()
    if ro_max > 0:
        few_ops = 1.0 - (remaining_ops / ro_max)
    else:
        few_ops = np.ones(n, dtype=np.float64)

    # Composite score: primary is urgency * blocking, secondary is few_ops
    # This non-linear product emphasizes jobs that are BOTH urgent AND short
    # on their machine, which are the best candidates to start immediately.
    primary = urgency * blocking
    combined = 0.85 * primary + 0.15 * few_ops

    best_idx = int(np.argmax(combined))

    if _check_deadline():
        return 0

    return best_idx
# EVOLVE-BLOCK-END
