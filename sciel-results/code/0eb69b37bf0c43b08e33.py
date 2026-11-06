"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

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
            # Use soft-min via log-sum-exp for a smooth surrogate of min duration
            # This accounts for the distribution of durations, not just the minimum
            durations = list(machine_options.values())
            if not durations:
                min_duration = 1.0
                best_machine_local = 0
            else:
                # Hard min for machine selection
                min_duration = min(durations)
                best_machine_local = list(machine_options.keys())[durations.index(min_duration)]

                if min_duration <= 0:
                    min_duration = 1.0

                # Soft-min: -ln(sum(exp(-d_i / tau))) * tau
                # Small tau -> approaches hard min; larger tau -> smoother
                # We use tau=2.0 as a balance
                tau = 2.0
                exp_vals = [math.exp(-d / tau) for d in durations]
                sum_exp = sum(exp_vals)
                if sum_exp > 0:
                    soft_min = -tau * math.log(sum_exp)
                else:
                    soft_min = min_duration

                # Blend soft-min and hard-min for efficiency score
                # soft_min >= min_duration always
                # Use a blend that's closer to hard-min but smoothed
                blended_duration = 0.7 * min_duration + 0.3 * soft_min
                if blended_duration <= 0:
                    blended_duration = 1.0

                eff_score = 1.0 / blended_duration

            # Refined Heuristic:
            # 1. Criticality: Normalized remaining work (0 to 1).
            # 2. Efficiency: Inverse of blended duration (shorter is better).
            # 3. Tail Reduction: Bonus for jobs with fewer remaining operations.
            # 4. Constraint penalty: operations with fewer machine options are more constrained

            # Normalize remaining work
            norm_work = remaining_work / max_remaining_work

            # Tail score: Jobs with fewer remaining ops should be prioritized to finish them.
            if remaining_ops > 0:
                tail_score = 1.0 / remaining_ops
            else:
                tail_score = 0.0

            # Constraint score: fewer machine options = more constrained = higher priority
            n_machines = len(machine_options)
            if n_machines > 0:
                constraint_score = 1.0 / n_machines
            else:
                constraint_score = 0.0

            # Combined score with adjusted weights:
            # Higher weight on criticality and efficiency, moderate on tail, small on constraint
            score = 0.45 * norm_work + 0.30 * eff_score + 0.15 * tail_score + 0.10 * constraint_score

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_local

    # Return the index and the explicitly selected best machine
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END
