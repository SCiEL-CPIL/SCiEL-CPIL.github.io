import numpy as np
import time as _time_module

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling (makespan minimization).
    
    Uses a combined scoring approach:
    - Urgency: remaining_ops (more remaining ops = more critical)
    - Speed: 1/min_duration (faster completion)
    - Contention avoidance: prefer machines with lower total load
    - Secondary: lower remaining_work (job closer to completion)
    
    The score balances urgency and speed while penalizing operations
    that would go to heavily-loaded machines.
    """
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)
    
    # First pass: compute machine loads (sum of durations per machine)
    # to estimate contention
    machine_loads = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for m, dur in machine_options.items():
            if m not in machine_loads:
                machine_loads[m] = 0.0
            machine_loads[m] += dur

    # Compute average load per machine for normalization
    if machine_loads:
        avg_load = sum(machine_loads.values()) / len(machine_loads)
    else:
        avg_load = 1.0

    best_index = 0
    best_score = -float('inf')
    best_secondary = float('inf')

    epsilon = 1e-9
    beta = 10.0  # Temperature for soft-min (higher = sharper)

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            soft_min_duration = 1.0
            contention_penalty = 0.0
        else:
            durations = list(machine_options.values())
            if not durations:
                soft_min_duration = 1.0
                contention_penalty = 0.0
            else:
                # Soft-min via log-sum-exp: smooth approximation of min
                # soft_min = -log(sum(exp(-d/beta)))/beta
                min_d = min(durations)
                if min_d <= 0:
                    min_d = 1e-9
                
                # Compute soft-min using log-sum-exp for numerical stability
                neg_d_over_beta = [-d / beta for d in durations]
                max_neg = max(neg_d_over_beta)
                sum_exp = sum(np.exp(x - max_neg) for x in neg_d_over_beta)
                soft_min_duration = -(max_neg + np.log(sum_exp)) * beta
                if soft_min_duration <= 0:
                    soft_min_duration = 1e-9
                
                # Find the machine with minimum duration and its load
                min_machine_load = float('inf')
                for m, d in machine_options.items():
                    if d == min(durations):
                        ml = machine_loads.get(m, 0.0)
                        if ml < min_machine_load:
                            min_machine_load = ml
                
                # Contention penalty: higher load on the best machine → lower score
                # Normalize by average load to keep scale consistent
                if avg_load > 0:
                    contention_ratio = min_machine_load / avg_load
                else:
                    contention_ratio = 1.0
                
                # Penalize high contention (ratio > 1 means above average load)
                contention_penalty = max(0.0, contention_ratio - 1.0) * 0.3

        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None:
            remaining_work = 0.0

        # Critical path slack: total remaining work minus current time
        # This estimates how much time is left for this job
        critical_path_slack = remaining_work - time if time is not None else remaining_work
        if critical_path_slack <= 0:
            critical_path_slack = 1e-9

        # Primary score: urgency × speed, reduced by contention
        # Use critical path slack and soft-min duration
        # Higher urgency and lower duration → higher base score
        base_score = remaining_ops / soft_min_duration
        
        # Apply contention penalty as a multiplicative discount
        # This keeps the score positive but reduces it for congested machines
        score = base_score * (1.0 - contention_penalty)

        # Secondary: prefer lower remaining_work (job closer to done)
        secondary = remaining_work

        if score > best_score + epsilon:
            best_score = score
            best_secondary = secondary
            best_index = i
        elif abs(score - best_score) <= epsilon:
            if secondary < best_secondary:
                best_secondary = secondary
                best_index = i

    return int(best_index)
