"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard variables
_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _time_exceeded():
    """Check if we are approaching the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def dispatch(ready_operations, time):
    # Explicit deadline guard
    if _time_exceeded():
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0

    best_score = -1.0
    best_index = 0
    best_machine = None

    # Pre-compute normalization constants
    max_remaining_work = max(op.get("remaining_work", 0.0) for op in ready_operations)
    if max_remaining_work <= 0:
        max_remaining_work = 1.0

    max_remaining_ops = max(op.get("remaining_ops", 0) for op in ready_operations)
    if max_remaining_ops <= 0:
        max_remaining_ops = 1

    # Find the maximum number of machine options for normalization
    max_machine_options = max(len(op.get("machine_options", {})) for op in ready_operations)
    if max_machine_options <= 0:
        max_machine_options = 1

    for i, op in enumerate(ready_operations):
        # Check deadline again in case of extremely large input (unlikely here)
        if _time_exceeded():
            break

        remaining_work = op.get("remaining_work", 0.0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)

        if not machine_options:
            score = 0.0
            best_machine_local = 0
        else:
            # Find the machine with the minimum duration for this operation
            min_duration = float('inf')
            best_machine_local = next(iter(machine_options.keys()))

            for m, dur in machine_options.items():
                if dur < min_duration:
                    min_duration = dur
                    best_machine_local = m

            if min_duration <= 0:
                min_duration = 1.0

            # Refined Heuristic:
            # 1. Criticality: Normalized remaining work (0 to 1).
            # 2. Efficiency: Inverse of min duration (shorter is better).
            # 3. Tail Reduction: Bonus for jobs with fewer remaining operations.
            # 4. Constraint Tightness: Bonus for operations with fewer machine options.

            # Normalize remaining work
            norm_work = remaining_work / max_remaining_work

            # Efficiency score: 1/duration.
            # If duration is 1, score is 1. If duration is 10, score is 0.1.
            # This heavily favors short processing times on the best machine.
            eff_score = 1.0 / min_duration

            # Tail score: Jobs with fewer remaining ops should be prioritized to finish them.
            # If remaining_ops is 1, tail_score is 1.0. If 10, 0.1.
            if remaining_ops > 0:
                tail_score = 1.0 / remaining_ops
            else:
                tail_score = 0.0

            # Constraint tightness score: Operations with fewer machine options are more constrained
            # and should be scheduled earlier to prevent future starvation.
            # If an operation has only 1 machine option, it's highly constrained.
            # If it has 5 options, it's less constrained.
            # We want higher score for fewer options.
            # Normalize: (max_options - current_options) / (max_options - 1)
            # If max_options is 1, this is 0/0, so we handle it.
            if max_machine_options > 1:
                constraint_score = (max_machine_options - len(machine_options)) / (max_machine_options - 1)
            else:
                constraint_score = 0.0

            # Combined score:
            # We want to balance completing high-work jobs vs short jobs vs finishing jobs vs constrained jobs.
            # Weights:
            # - norm_work: 0.4 (High work jobs are critical)
            # - eff_score: 0.3 (Short jobs free up machines quickly)
            # - tail_score: 0.15 (Finishing jobs reduces active count)
            # - constraint_score: 0.15 (Prevent starvation of constrained ops)

            score = 0.4 * norm_work + 0.3 * eff_score + 0.15 * tail_score + 0.15 * constraint_score

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_local

    # Return the index and the explicitly selected best machine
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
