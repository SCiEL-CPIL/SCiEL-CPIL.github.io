"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Returns the index of the ready operation to start next.
    
    Strategy: Combine multiple heuristics:
    1. Critical Path Method: Prioritize operations with longest remaining critical path
    2. Machine load balancing: Prefer less loaded machines
    3. Urgency: Consider remaining work vs current time
    4. Shortest processing time on the chosen machine as tiebreaker
    
    We use a weighted combination:
    - Primary: Maximize remaining critical path (remaining_work of the job)
    - Secondary: Minimize the minimum duration among machine options (shorter ops first)
    - Tertiary: Balance machine loads (prefer machines with less total assigned work)
    """
    if not ready_operations:
        return 0
    
    n = len(remaining_ops_list) if False else len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Deadline guard
    start_time = _time.time()
    deadline = start_time + 5.0  # 5 second safety margin
    
    # Precompute some values
    scores = [0.0] * n
    
    # First pass: compute base scores based on remaining work and urgency
    max_remaining_work = 0
    min_duration = float('inf')
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Get minimum processing time among machine options
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = list(machine_options.values())
            min_dur = min(durations)
            max_dur = max(durations)
            avg_dur = sum(durations) / len(durations)
        else:
            min_dur = 1
            max_dur = 1
            avg_dur = 1
        
        max_remaining_work = max(max_remaining_work, remaining_work)
        min_duration = min(min_duration, min_dur)
        
        # Base score: prioritize jobs with more remaining work
        # Normalize remaining work
        if max_remaining_work > 0:
            work_score = remaining_work / max_remaining_work
        else:
            work_score = 0
        
        # Urgency score: how urgent is this operation?
        # Higher remaining work relative to current time means more urgent
        if remaining_ops > 0:
            urgency = remaining_work / (remaining_ops * max(1, time + 1))
        else:
            urgency = remaining_work
        
        # Shorter processing time is generally better for makespan
        # Invert: shorter = higher score
        if min_duration > 0:
            duration_score = 1.0 / (min_dur + 1.0)
        else:
            duration_score = 1.0
        
        # Weighted combination
        scores[i] = 0.5 * work_score + 0.3 * urgency + 0.2 * duration_score
    
    # Check deadline
    if _time.time() > deadline - 1.0:
        # Return best found so far
        best_idx = 0
        for i in range(1, n):
            if scores[i] > scores[best_idx]:
                best_idx = i
        return int(best_idx)
    
    # Second pass: consider machine load balancing
    # Estimate machine loads from the ready operations' machine options
    machine_load = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine, duration in machine_options.items():
            if machine not in machine_load:
                machine_load[machine] = 0
            machine_load[machine] += duration
    
    # Normalize machine loads
    max_load = max(machine_load.values()) if machine_load else 1
    if max_load > 0:
        for m in machine_load:
            machine_load[m] /= max_load
    
    # Adjust scores based on machine load balance
    adjusted_scores = [0.0] * n
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            adjusted_scores[i] = scores[i]
            continue
        
        # Find the machine with minimum load among options
        min_load_machine = None
        min_load = float('inf')
        for machine, duration in machine_options.items():
            load = machine_load.get(machine, 0)
            if load < min_load:
                min_load = load
                min_load_machine = machine
        
        # Load balance bonus: prefer operations that can use less loaded machines
        # Lower load = higher bonus
        load_bonus = 1.0 - min_load  # 0 to 1, higher is better (less loaded)
        
        adjusted_scores[i] = scores[i] * 0.8 + load_bonus * 0.2
    
    # Check deadline again
    if _time.time() > deadline - 0.5:
        best_idx = 0
        for i in range(1, n):
            if adjusted_scores[i] > adjusted_scores[best_idx]:
                best_idx = i
        return int(best_idx)
    
    # Third pass: tiebreaker - shortest processing time on preferred machine
    final_scores = [0.0] * n
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            final_scores[i] = adjusted_scores[i]
            continue
        
        # Get the minimum duration (shortest processing time option)
        min_dur = min(machine_options.values())
        
        # Slight bonus for shorter operations
        duration_bonus = 1.0 / (min_dur + 0.1)
        
        final_scores[i] = adjusted_scores[i] * 0.95 + duration_bonus * 0.05
    
    # Select the operation with highest score
    best_idx = 0
    for i in range(1, n):
        if final_scores[i] > final_scores[best_idx]:
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
