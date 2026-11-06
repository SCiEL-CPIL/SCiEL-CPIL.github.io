"""Open-shop scheduling dispatch rule: minimize makespan via combined heuristics."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_MARGIN = 0.5  # seconds to stay under the 120s wall-clock limit
_START_TIME = _time.monotonic()


def _check_deadline():
    """Return True if we are approaching the wall-clock deadline."""
    return (_time.monotonic() - _START_TIME) > (120.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Combine multiple heuristics to estimate which choice leads to the
    lowest makespan:
    1. Remaining work (total processing time left for the job)
    2. Remaining ops count (more ops = more critical path)
    3. Minimum possible completion time if we pick this op (duration on best machine)
    4. Machine load balancing: prefer machines that are less loaded (fewer other ops)
    
    We score each ready operation with a weighted combination and pick the best.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _check_deadline():
        # Fallback: pick max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Pre-compute features for each ready operation
    # Feature 1: remaining_work (higher = more critical)
    # Feature 2: remaining_ops (higher = more critical path)
    # Feature 3: min duration across machine_options (shorter = less blocking)
    # Feature 4: max duration across machine_options (longer = more risk)
    # Feature 5: number of machine options (more flexibility)
    
    rem_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    rem_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    min_durs = np.zeros(n, dtype=np.float64)
    max_durs = np.zeros(n, dtype=np.float64)
    n_machines = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        if durations:
            min_durs[i] = min(durations)
            max_durs[i] = max(durations)
            n_machines[i] = len(durations)
        else:
            min_durs[i] = 0.0
            max_durs[i] = 0.0
            n_machines[i] = 0.0

    # Normalize features to [0, 1] range for weighted combination
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    nw = normalize(rem_works)
    no = normalize(rem_ops)
    nd_min = normalize(min_durs)
    nd_max = normalize(max_durs)
    nm = normalize(n_machines)

    # Scoring: We want to prioritize operations that:
    # - Have high remaining work (critical jobs)
    # - Have many remaining ops (long critical path)
    # - Have short minimum duration (can finish quickly, reducing makespan)
    # - Have many machine options (flexibility)
    #
    # The key insight for open-shop: we want to keep all machines busy and
    # complete jobs that have the most work left. A good proxy is to prioritize
    # jobs with high remaining work, but also consider that starting a short
    # operation frees up the job sooner.
    
    # Weighted score: higher is better (we pick argmax)
    w_work = 0.35
    w_ops = 0.25
    w_min_dur = 0.15  # prefer shorter min duration
    w_max_dur = 0.10  # prefer shorter max duration (less risk)
    w_flex = 0.15     # prefer more machine options

    score = (
        w_work * nw +
        w_ops * no +
        w_min_dur * (1.0 - nd_min) +  # invert: shorter is better
        w_max_dur * (1.0 - nd_max) +  # invert: shorter is better
        w_flex * nm
    )

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
