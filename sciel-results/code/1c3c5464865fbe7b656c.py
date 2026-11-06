import numpy as np
import time as _time_module

# EVOLVE-BLOCK-START
_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule: Select the operation that maximizes critical-path urgency.
    
    Primary score: remaining_work / soft_min_duration
    - remaining_work captures total remaining processing time (critical path length)
    - soft_min_duration is a smooth surrogate for the best available machine speed
    
    This prioritizes operations with high remaining work that can be processed
    quickly, reducing overall makespan by clearing long chains early.
    
    Secondary tie-breaker: lower remaining_ops (closer to completion) preferred.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're running long, return quickly
    if _time_module.perf_counter() > _DEADLINE:
        return 0

    best_index = 0
    best_score = -float('inf')
    best_secondary = float('inf')

    n_ops = len(ready_operations)

    for i, op in enumerate(ready_operations):
        # Calculate soft-min duration from machine options
        machine_options = op.get("machine_options", {})
        if not machine_options:
            # Fallback if no options (shouldn't happen)
            soft_min_duration = 1.0
        else:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            if durations.size == 0:
                soft_min_duration = 1.0
            else:
                # Clip to avoid log(0)
                durations = np.clip(durations, 1e-9, None)
                # Soft-min via log-sum-exp: -log(sum(exp(-x/tau))) * tau
                # As tau -> 0, this approaches min(x)
                tau = 0.3  # Lower temperature for sharper min approximation
                neg_durations = -durations / tau
                # Numerically stable log-sum-exp
                max_neg = np.max(neg_durations)
                log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_durations - max_neg)))
                soft_min_duration = -tau * log_sum_exp
                if soft_min_duration <= 0:
                    soft_min_duration = 1e-9

        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None:
            remaining_work = 0.0
        if remaining_work <= 0:
            remaining_work = 1e-9

        # Calculate primary score
        # Higher is better: prioritize high remaining work that can be done quickly
        score = remaining_work / soft_min_duration

        # Decision: use primary score, with secondary as tie-breaker
        # Small epsilon to treat near-equal scores as ties
        epsilon = 1e-6
        if score > best_score + epsilon:
            best_score = score
            best_secondary = remaining_ops
            best_index = i
        elif abs(score - best_score) <= epsilon:
            # Tie-break: prefer lower remaining_ops (closer to completion)
            if remaining_ops < best_secondary:
                best_score = score
                best_secondary = remaining_ops
                best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
