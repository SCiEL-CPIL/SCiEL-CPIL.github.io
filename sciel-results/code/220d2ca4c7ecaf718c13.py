"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: keep explicit wall-clock budget.
    # The evaluator kills after 150s; we want to return well before that.
    _start = _time.perf_counter()
    _deadline = _start + 115.0  # 115s safety margin

    best_score = -1.0
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            score = 0.0
            best_machine_candidate = None
        else:
            # Choose the machine with minimum duration (fastest completion)
            min_duration = min(machine_options.values())
            best_machine_candidate = min(machine_options, key=machine_options.get)

            if min_duration <= 0:
                min_duration = 1.0

            # Heuristic components:
            # 1. Remaining work / min_duration: throughput/urgency ratio
            # 2. Bonus for fewer remaining operations (closer to completion)
            # 3. Penalty for long min_duration (blocks machine longer)
            
            # Primary: work-to-time ratio
            work_ratio = remaining_work / min_duration
            
            # Secondary: prioritize operations with fewer remaining steps
            # (completing a job frees up its remaining ops sooner)
            step_bonus = 1.0 / (1.0 + remaining_ops) if remaining_ops > 0 else 1.0
            
            # Tertiary: slight preference for shorter operations
            duration_penalty = 1.0 / (1.0 + min_duration * 0.1)
            
            score = work_ratio * (0.8 + 0.2 * step_bonus) * duration_penalty

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_candidate

        # Check deadline periodically
        if i % 100 == 0 and _time.perf_counter() > _deadline:
            break

    return int(best_index)
# EVOLVE-BLOCK-END
