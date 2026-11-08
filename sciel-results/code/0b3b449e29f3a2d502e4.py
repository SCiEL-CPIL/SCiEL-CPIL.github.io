"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track per-call time to avoid pathological cases, but dispatch is O(n) so this is mostly safety.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin


def _is_deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if _is_deadline_exceeded():
        # Fallback: pick the operation with the most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)

    # Fast path: if only one operation, return it
    if n == 1:
        return 0

    # Extract features for all operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    min_durations = np.empty(n, dtype=np.float64)
    max_durations = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            vals = list(machine_options.values())
            min_durations[i] = min(vals)
            max_durations[i] = max(vals)
        else:
            min_durations[i] = 0.0
            max_durations[i] = 0.0

    # Compute scores using vectorized operations
    # WSPT-like ratio: remaining_work / (min_duration + epsilon)
    wspt_ratio = remaining_works / (min_durations + 1e-9)

    # Urgency: jobs with fewer remaining operations are closer to completion
    urgency = 1.0 / (remaining_ops + 1.0)

    # Critical path weight: total remaining work
    critical_weight = remaining_works

    # Shortest processing time bonus
    spt_bonus = 1.0 / (min_durations + 1.0)

    # Machine flexibility: prefer operations with more machine options (more flexible)
    # This can help avoid bottlenecks
    machine_counts = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    flexibility = machine_counts / (max(machine_counts) + 1e-9) if max(machine_counts) > 0 else np.ones(n)

    # Duration variance: prefer operations with less variance in machine durations
    # (more predictable)
    if n > 1:
        durations_var = np.array([
            np.var(list(op["machine_options"].values())) if len(op["machine_options"]) > 1 else 0.0
            for op in ready_operations
        ], dtype=np.float64)
        # Lower variance is better, so we use negative variance as a bonus
        var_bonus = -durations_var
    else:
        var_bonus = np.zeros(n)

    # Combine features with weights
    # Tuned weights: emphasize WSPT and critical path, with urgency and SPT as secondary
    w_wspt = 0.30
    w_critical = 0.30
    w_urgency = 0.20
    w_spt = 0.10
    w_flex = 0.05
    w_var = 0.05

    scores = (
        w_wspt * wspt_ratio
        + w_critical * critical_weight
        + w_urgency * urgency * remaining_works
        + w_spt * spt_bonus
        + w_flex * flexibility * remaining_works
        + w_var * var_bonus
    )

    # Return index of the operation with the highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
