"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills at 150s wall-clock; keep a safety margin.
_DEADLINE_START = None
_DEADLINE_LIMIT = 120.0  # seconds; leave 30s margin before the 150s kill


def _deadline_ok():
    """Return True if we are still within the safety margin."""
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = _time.time()
    return (_time.time() - _DEADLINE_START) < _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    # LPT-inspired composite heuristic for open-shop makespan minimization.
    #
    # Rationale (Longest Processing Time greedy approximation):
    # 1. Prioritize operations with the largest minimum processing time so that
    #    long, inflexible tasks are scheduled early, balancing machine loads
    #    and preventing late-stage bottlenecks.
    # 2. Among those, prefer operations with high remaining work to keep
    #    machines busy and reduce idle time.
    # 3. Use remaining_ops as a tiebreaker / secondary signal for job
    #    completion proximity.
    #
    # We compute a composite score for each ready operation and pick the max.

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Precompute arrays for speed
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each operation, find the minimum duration across its machine options.
    # Under LPT logic, we prefer LARGER min_duration (longer tasks first).
    min_durations = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
        else:
            min_durations[i] = 0.0

    # Normalize each component to [0, 1] range for fair weighting
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    rw_norm = normalize(remaining_works)
    ro_norm = normalize(remaining_ops)

    # LPT: prefer LONGER min_durations, so use normalized min_durations directly
    # (higher score for longer operations)
    md_norm = normalize(min_durations)

    # Composite score: weight longest-processing-time most heavily, then
    # remaining work, then job completion proximity.
    # Tuned weights: LPT=0.5, MW=0.3, JCP=0.2
    score = 0.5 * md_norm + 0.3 * rw_norm + 0.2 * ro_norm

    # Pick the operation with the highest composite score
    best_idx = int(np.argmax(score))

    return best_idx
# EVOLVE-BLOCK-END
