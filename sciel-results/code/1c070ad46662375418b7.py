"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

# Global state for load balancing
# Assuming 5 machines based on task description
_MACHINE_COUNT = 5
_machine_load = np.zeros(_MACHINE_COUNT)  # busy-until time per machine
_last_seen_time = -1.0

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Args:
        ready_operations: List of dicts with job, op, machine_options, remaining_ops, remaining_work
        time: Current simulation time
        
    Returns:
        Index of the operation to start, or (index, machine) tuple.
    """
    global _machine_load, _last_seen_time

    if not ready_operations:
        return 0

    # Deadline check
    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: Simple Most Remaining Work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Detect new instance: if time resets, reset load counters
    if time < _last_seen_time:
        _machine_load = np.zeros(_MACHINE_COUNT)
    _last_seen_time = time

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            # Pick the machine that gives earliest completion
            best_machine = None
            best_completion = float('inf')
            for m, d in machine_options.items():
                if m < _MACHINE_COUNT:
                    start = max(time, _machine_load[m])
                    completion = start + d
                else:
                    start = time
                    completion = time + d
                if completion < best_completion:
                    best_completion = completion
                    best_machine = m
            if best_machine is not None and best_machine < _MACHINE_COUNT:
                _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    # Flexibility: 1 / number of machine options.
    # Less options -> Higher priority (constrained)
    flex_scores = np.array([
        1.0 / len(op["machine_options"]) if op["machine_options"] else 0.0
        for op in ready_operations
    ])

    # Compute earliest completion time for each operation
    earliest_completions = np.zeros(n)
    min_durations = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
            best_comp = float('inf')
            for m, d in mo.items():
                if m < _MACHINE_COUNT:
                    start = max(time, _machine_load[m])
                    completion = start + d
                else:
                    start = time
                    completion = time + d
                if completion < best_comp:
                    best_comp = completion
            earliest_completions[i] = best_comp
        else:
            min_durations[i] = float('inf')
            earliest_completions[i] = time + op["remaining_work"]

    # Look-ahead bottleneck heuristic: score by minimum completion time
    # considering machine load, with flexibility as tie-breaker
    # Lower score = better (earlier completion)
    
    # Compute soft-min completion time for each operation
    # This gives a smooth surrogate for the min completion across machines
    beta = 2.0  # Sharpness parameter for soft-min
    soft_min_completions = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            completions = []
            for m, d in mo.items():
                if m < _MACHINE_COUNT:
                    start = max(time, _machine_load[m])
                    completion = start + d
                else:
                    start = time
                    completion = time + d
                completions.append(completion)
            
            # Use soft-min to get a smooth estimate of best completion
            if len(completions) == 1:
                soft_min_completions[i] = completions[0]
            else:
                c = np.array(completions, dtype=np.float64)
                c_min = np.min(c)
                scaled = -beta * (c - c_min)
                max_scaled = np.max(scaled)
                if max_scaled > 700:
                    max_scaled = 700
                sum_exp = np.sum(np.exp(scaled - max_scaled))
                if sum_exp == 0:
                    soft_min_completions[i] = c_min
                else:
                    log_sum = max_scaled + np.log(sum_exp)
                    soft_min_completions[i] = c_min + log_sum / beta
        else:
            soft_min_completions[i] = time + op["remaining_work"]

    # Primary criterion: earliest soft-min completion time (machine-centric load balancing)
    min_completion = np.min(soft_min_completions)
    completion_mask = soft_min_completions <= min_completion + 1e-6
    completion_candidates = [i for i in range(n) if completion_mask[i]]
    
    if len(completion_candidates) == 1:
        best_idx = completion_candidates[0]
    else:
        # Secondary criterion: among tied completions, prefer more constrained (higher flex)
        cand_flex = flex_scores[completion_candidates]
        max_flex_cand = np.max(cand_flex)
        flex_mask = cand_flex >= max_flex_cand - 1e-9
        flex_candidates = [completion_candidates[i] for i in range(len(completion_candidates)) if flex_mask[i]]
        
        if len(flex_candidates) == 1:
            best_idx = flex_candidates[0]
        else:
            # Tertiary criterion: prefer more remaining work
            cand_work = remaining_works[flex_candidates]
            max_work_cand = np.max(cand_work)
            work_mask_cand = cand_work >= max_work_cand - 1e-9
            work_candidates = [flex_candidates[i] for i in range(len(flex_candidates)) if work_mask_cand[i]]
            
            if len(work_candidates) == 1:
                best_idx = work_candidates[0]
            else:
                # Final tie-break: min duration
                cand_durations = min_durations[work_candidates]
                best_idx = work_candidates[int(np.argmin(cand_durations))]

    # Select machine for the chosen operation: minimize completion time
    # with mild congestion awareness to balance load
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx

    best_machine = None
    best_score = float('inf')
    
    # Congestion penalty: discourage assigning to already-busy machines
    congestion_alpha = 0.05
    
    for m, d in machine_options.items():
        if m < _MACHINE_COUNT:
            start = max(time, _machine_load[m])
            completion = start + d
            # Penalty proportional to how much the machine is already loaded
            congestion_penalty = congestion_alpha * max(0.0, _machine_load[m] - time)
            score = completion + congestion_penalty
        else:
            start = time
            completion = time + d
            score = completion

        if score < best_score:
            best_score = score
            best_machine = m

    # Update machine load
    if best_machine is not None and best_machine < _MACHINE_COUNT:
        _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
