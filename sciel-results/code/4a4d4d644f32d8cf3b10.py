import numpy as np
import time as _time_module

# EVOLVE-BLOCK-START
def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Uses a composite heuristic that balances:
    1. Remaining work (to prioritize critical jobs)
    2. Immediate processing time (to keep flow)
    3. Machine load considerations
    """
    if not ready_operations:
        return 0

    # Deadline guard: track elapsed time
    start_time = _time_module.perf_counter()
    deadline = start_time + 1.0  # 1 second safety margin per dispatch call

    best_index = 0
    best_score = -float('inf')

    for i, op in enumerate(ready_operations):
        # Check deadline
        if _time_module.perf_counter() > deadline:
            break

        # Get machine options
        machine_options = op.get("machine_options", {})
        if not machine_options:
            min_duration = 1.0
            avg_duration = 1.0
        else:
            durations = list(machine_options.values())
            min_duration = min(durations)
            avg_duration = sum(durations) / len(durations) if durations else 1.0
            if min_duration <= 0:
                min_duration = 1e-9
            if avg_duration <= 0:
                avg_duration = 1e-9

        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        remaining_work = op.get("remaining_work", 0)
        if remaining_work <= 0:
            remaining_work = min_duration

        # Composite score:
        # - Weighted by remaining ops (prioritize jobs with more work left)
        # - Inversely weighted by min duration (shorter jobs first)
        # - Additional term for remaining work to handle critical path
        #
        # Score = (remaining_ops * 1.0 + remaining_work * 0.1) / (min_duration * 1.0 + avg_duration * 0.5)
        #
        # This creates a balance between:
        # 1. Jobs with many remaining operations (to avoid blocking)
        # 2. Short immediate processing times (to keep flow)
        # 3. Total remaining work (critical path consideration)

        numerator = remaining_ops * 1.0 + remaining_work * 0.1
        denominator = min_duration * 1.0 + avg_duration * 0.5

        if denominator <= 0:
            denominator = 1e-9

        score = numerator / denominator

        if score > best_score:
            best_score = score
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
