"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

# Global state for load balancing
_MACHINE_COUNT = 5
_machine_busy_until = np.zeros(_MACHINE_COUNT)  # Time when machine becomes free
_last_seen_time = -1.0

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    Uses look-ahead bottleneck heuristic: score each operation by its
    minimum achievable completion time considering machine load.
    """
    global _machine_busy_until, _last_seen_time

    if not ready_operations:
        return 0

    # Deadline check
    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Detect new instance
    if time < _last_seen_time:
        _machine_busy_until = np.zeros(_MACHINE_COUNT)
    _last_seen_time = time

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            # Pick machine with earliest availability + duration
            best_m = None
            best_t = float('inf')
            for m, d in machine_options.items():
                avail = _machine_busy_until[m] if m < _MACHINE_COUNT else 0.0
                finish = max(time, avail) + d
                if finish < best_t:
                    best_t = finish
                    best_m = m
            if best_m is not None and best_m < _MACHINE_COUNT:
                _machine_busy_until[best_m] = max(time, _machine_busy_until[best_m]) + machine_options[best_m]
            return (0, best_m)
        return 0

    # For each ready operation, compute the minimum completion time
    # considering all eligible machines and current load
    min_completion_times = np.zeros(n)
    best_machines = [None] * n

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            min_completion_times[i] = float('inf')
            continue

        best_finish = float('inf')
        best_m = None
        for m, d in machine_options.items():
            avail = _machine_busy_until[m] if m < _MACHINE_COUNT else 0.0
            finish = max(time, avail) + d
            if finish < best_finish:
                best_finish = finish
                best_m = m

        min_completion_times[i] = best_finish
        best_machines[i] = best_m

    # Primary score: earliest completion time (minimize)
    # Secondary: prefer operations with more remaining work (urgency)
    # Tertiary: prefer operations with fewer machine options (constraint)
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    flex_scores = np.array([
        1.0 / len(op["machine_options"]) if op["machine_options"] else 0.0
        for op in ready_operations
    ])

    # Normalize
    max_ct = np.max(min_completion_times)
    min_ct = np.min(min_completion_times)
    max_work = np.max(remaining_works)
    max_flex = np.max(flex_scores)

    # Score: we want to MINIMIZE completion time, so invert
    if max_ct > min_ct:
        norm_ct = (min_completion_times - min_ct) / (max_ct - min_ct)  # 0 = best, 1 = worst
    else:
        norm_ct = np.zeros(n)

    if max_work > 0:
        norm_work = remaining_works / max_work
    else:
        norm_work = np.zeros(n)

    if max_flex > 0:
        norm_flex = flex_scores / max_flex
    else:
        norm_flex = np.zeros(n)

    # Weighted score: lower completion time is best (primary),
    # higher remaining work is more urgent (secondary),
    # higher inflexibility is more constrained (tertiary)
    w_ct = 1.0
    w_work = 0.3
    w_flex = 0.2

    scores = -w_ct * norm_ct + w_work * norm_work + w_flex * norm_flex

    best_idx = int(np.argmax(scores))

    # Machine selection for the chosen operation
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx

    # Pick machine that gives earliest finish, with tie-breaking by less future congestion
    best_m = None
    best_finish = float('inf')

    for m, d in machine_options.items():
        avail = _machine_busy_until[m] if m < _MACHINE_COUNT else 0.0
        finish = max(time, avail) + d
        if finish < best_finish:
            best_finish = finish
            best_m = m
        elif abs(finish - best_finish) < 1e-9:
            # Tie: prefer machine with lower current load
            if m < _MACHINE_COUNT:
                if best_m is not None and best_m < _MACHINE_COUNT:
                    if _machine_busy_until[m] < _machine_busy_until[best_m]:
                        best_m = m

    if best_m is not None and best_m < _MACHINE_COUNT:
        _machine_busy_until[best_m] = best_finish

    return (best_idx, best_m)
# EVOLVE-BLOCK-END
