"""Job-shop scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


def _elapsed():
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    # Deadline guard
    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # For each operation, compute a score that balances:
    # 1. Remaining work (longer jobs should be started earlier - similar to seed)
    # 2. Machine-contention avoidance: prefer operations that can run on less-loaded machines
    # 3. Urgency: operations with fewer remaining ops are closer to completion

    scores = [0.0] * n

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            scores[i] = remaining_work
            continue

        # Get durations for all available machines
        durations = list(machine_options.values())
        min_dur = min(durations)
        max_dur = max(durations)
        mean_dur = sum(durations) / len(durations)

        # Machine-contention score: ratio of min duration to total duration
        # Lower ratio means the operation is more flexible (can pick a shorter-duration machine)
        # Higher ratio means it's more constrained
        total_dur = sum(durations)
        if total_dur > 0:
            flexibility_ratio = min_dur / total_dur
        else:
            flexibility_ratio = 1.0

        # Number of machine options: more options = less contention
        n_machines = len(machine_options)

        # Score components:
        # - remaining_work: prioritize jobs with more work (like seed)
        # - flexibility: prioritize operations with more machine options and lower contention
        # - urgency: prioritize operations closer to completion (fewer remaining ops)

        # Normalize components to avoid scale issues
        # Use a weighted combination that emphasizes remaining work but incorporates machine flexibility
        work_component = remaining_work
        flexibility_component = n_machines * (1.0 - flexibility_ratio)  # higher = more flexible
        urgency_component = 1.0 / max(remaining_ops, 1)

        # Weight: work is primary, flexibility secondary, urgency tertiary
        score = work_component + 0.3 * flexibility_component + 0.1 * urgency_component
        scores[i] = score

    # Return index of operation with highest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
