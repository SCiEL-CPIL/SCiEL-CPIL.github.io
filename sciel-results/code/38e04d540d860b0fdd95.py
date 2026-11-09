"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import heapq

# Stateful tracking for the Greedy Local Search
# We maintain the current estimated free time for each machine.
# This allows us to simulate the impact of each candidate dispatch.
_machine_free_times = {}
_start_time = None
_dispatch_count = 0

def dispatch(ready_operations, time):
    global _start_time, _dispatch_count
    
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard: If we are running too long, fall back to a very fast heuristic
    # to ensure we finish within the 120s limit.
    if _time.time() - _start_time > 105:
        # Fallback: SPT on the machine that is free earliest
        best_idx = 0
        best_key = (float('inf'), float('inf'))
        for i, op in enumerate(ready_operations):
            options = op["machine_options"]
            if not options:
                continue
            min_completion = float('inf')
            for m, dur in options.items():
                free_t = _machine_free_times.get(m, 0)
                completion = max(time, free_t) + dur
                if completion < min_completion:
                    min_completion = completion
            if (min_completion, 0) < best_key:
                best_key = (min_completion, 0)
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0

    _dispatch_count += 1
    
    # Strategy: Greedy Local Search (Depth-1 Lookahead)
    # 1. For each ready operation, determine the best machine to assign it to 
    #    (minimizing completion time for that specific op).
    # 2. Simulate assigning that operation to that machine.
    # 3. Calculate the resulting "makespan estimate" (max over all machines of their new free times).
    # 4. Select the operation that results in the lowest makespan estimate.
    # 5. If there's a tie, prefer the operation with higher remaining work (to keep critical jobs moving).
    
    candidates = []
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        job_id = op["job"]
        job_rem_work = op["remaining_work"]
        
        # Find the best machine for this specific operation (earliest completion)
        best_m_for_op = None
        min_completion_for_op = float('inf')
        
        for m, dur in options.items():
            free_t = _machine_free_times.get(m, 0)
            start_t = max(time, free_t)
            completion_t = start_t + dur
            
            if completion_t < min_completion_for_op:
                min_completion_for_op = completion_t
                best_m_for_op = m

        if best_m_for_op is None:
            continue
            
        # Simulate the assignment
        # Create a copy of the current machine free times to avoid mutation during evaluation
        # Since n_machines is small (5), this is cheap.
        simulated_free_times = dict(_machine_free_times)
        
        # Update the selected machine's free time
        dur = op["machine_options"][best_m_for_op]
        start_t = max(time, simulated_free_times.get(best_m_for_op, 0))
        finish_t = start_t + dur
        simulated_free_times[best_m_for_op] = finish_t
        
        # Calculate the makespan estimate: max free time across all machines
        # Note: We must also consider the 'time' parameter, but since all free times 
        # are >= time (or we start at time), the max free time is the bottleneck.
        current_max_load = max(simulated_free_times.values()) if simulated_free_times else time
        
        # Score:
        # Primary: Minimize current_max_load (makespan estimate)
        # Secondary: Maximize job_rem_work (tie-breaker to prioritize critical jobs)
        
        # We want to minimize (current_max_load, -job_rem_work)
        # Or equivalently, maximize (-current_max_load, job_rem_work)
        
        candidates.append((current_max_load, -job_rem_work, i, best_m_for_op))
        
    if not candidates:
        return 0

    # Sort candidates by (makespan_estimate, -remaining_work)
    # We want the smallest makespan_estimate, and for ties, the largest remaining_work (smallest -remaining_work)
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    best_candidate = candidates[0]
    best_idx = best_candidate[2]
    best_machine = best_candidate[3]
    
    # Update global state with the chosen assignment
    sel_op = ready_operations[best_idx]
    dur = sel_op["machine_options"][best_machine]
    start_t = max(time, _machine_free_times.get(best_machine, 0))
    finish_t = start_t + dur
    _machine_free_times[best_machine] = finish_t
    
    # Return (index, machine) to ensure state consistency
    return (int(best_idx), best_machine)

# EVOLVE-BLOCK-END
