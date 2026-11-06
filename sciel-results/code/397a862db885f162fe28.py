"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    _start = _time.perf_counter()
    _deadline = _start + 115.0  # 115s safety margin (evaluator kills at 150s)

    best_score = -float('inf')
    best_index = 0

    for i, op in enumerate(ready_operations):
        # Deadline guard: if we're running out of time, return best so far
        if _time.perf_counter() > _deadline:
            break

        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            score = 0.0
        else:
            # Find the minimum duration among eligible machines (SPT on machine choice)
            min_duration = min(machine_options.values())

            if min_duration <= 0:
                min_duration = 1e-9

            # Composite heuristic combining:
            # 1) SPT: prefer shorter processing times (reduces makespan)
            # 2) FAS: prefer operations with fewer remaining ops (closer to completion)
            # 3) Work urgency: operations with more remaining work are more critical
            #
            # score = alpha / min_duration + beta / (remaining_ops + 1) + gamma * remaining_work / (min_duration * (remaining_ops + 1))
            #
            # The first term is pure SPT.
            # The second term favors jobs closer to completion (reduces tail).
            # The third term is a cross-term that rewards high-work, short-time, near-completion ops.

            spt_term = 1.0 / min_duration
            depth_term = 1.0 / (remaining_ops + 1)
            work_term = remaining_work / (min_duration * (remaining_ops + 1))

            # Weighted combination: SPT is strongest, depth next, work urgency last
            score = 1.0 * spt_term + 0.5 * depth_term + 0.1 * work_term

        if score > best_score:
            best_score = score
            best_index = i

    # Return index only (matching original interface)
    return int(best_index)
# EVOLVE-BLOCK-END
