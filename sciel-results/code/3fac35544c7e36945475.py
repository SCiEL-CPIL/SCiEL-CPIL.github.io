"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 140.0  # wall-clock safety margin before 150s kill
_START = _time.time()


def _remaining_time():
    return _DEADLINE - (_time.time() - _START)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    Returns an index (int) of the ready operation to start.
    Optionally could return (index, machine) but we stick with index for compatibility.
    """
    if not ready_operations:
        return 0

    # Fast path: if only one operation, return it
    if len(ready_operations) == 1:
        return 0

    # Check deadline
    if _remaining_time() < 0.5:
        # Fallback: pick the one with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)

    # Compute features for each ready operation
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # For each operation, compute the minimum duration across its machine options
    min_durations = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
        else:
            min_durations[i] = 1.0

    # Composite score: we want to prioritize operations that have:
    # 1. High remaining work (more urgent to finish)
    # 2. Fewer remaining ops (closer to completion, can reduce makespan by finishing jobs)
    # 3. Shorter minimum processing time (can be done quickly, reducing idle time)
    
    # Normalize features to [0, 1] range for combination
    rw_max = remaining_work.max() if remaining_work.max() > 0 else 1.0
    rw_min = remaining_work.min()
    rw_norm = (remaining_work - rw_min) / (rw_max - rw_min + 1e-10)

    ro_max = remaining_ops.max() if remaining_ops.max() > 0 else 1.0
    ro_min = remaining_ops.min()
    ro_norm = (remaining_ops - ro_min) / (ro_max - ro_min + 1e-10)

    md_max = min_durations.max() if min_durations.max() > 0 else 1.0
    md_min = min_durations.min()
    md_norm = (min_durations - md_min) / (md_max - md_min + 1e-10)

    # Weighted composite score
    # Higher score = more urgent
    # We weight remaining work heavily, then fewer remaining ops, then shorter duration
    w_rw = 0.5
    w_ro = 0.3
    w_md = 0.2

    # For remaining ops, fewer is better, so we invert
    # For min duration, shorter is better, so we invert
    score = w_rw * rw_norm + w_ro * (1.0 - ro_norm) + w_md * (1.0 - md_norm)

    # Select the operation with the highest composite score
    return int(np.argmax(score))
# EVOLVE-BLOCK-END
