"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Stateful tracking for Critical Path Lookahead Dispatch
_machine_free_times = {}
_start_time = None
_dispatch_count = 0
_last_time = -1
_last_instance_id = None

def dispatch(ready_operations, time):
    global _start_time, _dispatch_count, _last_time, _machine_free_times, _last_instance_id
    
    # Detect new instance (time goes backwards or significantly resets)
    if time < _last_time - 1e-6:
        _machine_free_times = {}
        _dispatch_count = 0
        _last_instance_id = None
    _last_time = time
    
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard: If we are running too long, fall back to a very fast heuristic
    if _time.time() - _start_time > 110:
        # Fallback: Shortest Processing Time (SPT) on the machine that is free earliest
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
    
    # Strategy: Critical Path Lookahead Dispatch
    # We model the job-shop as a dynamic bipartite graph between ready operations and machines.
    # We select the operation-machine pair that minimizes the estimated increase in the 
    # global critical path length.
    # 
    # The critical path length is approximated by the sum of the current operation's duration
    # and the maximum remaining work of the job it unblocks. This shifts the strategy from 
    # static priority scoring to a graph-theoretic lookahead that explicitly penalizes 
    # operations that extend the longest chain of dependencies.
    
    candidates = []
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        job_id = op["job"]
        job_rem_work = op["remaining_work"]
        
        # Find the machine assignment that minimizes the estimated critical path increase
        best_m_for_op = None
        min_critical_path_increase = float('inf')
        min_completion_for_op = float('inf')
        
        for m, dur in options.items():
            free_t = _machine_free_times.get(m, 0)
            start_t = max(time, free_t)
            completion_t = start_t + dur
            
            # Estimate the critical path increase:
            # 1. The operation itself adds 'dur' to the path.
            # 2. The job's remaining work (excluding current op) represents the future path length.
            #    We use 'remaining_work' as a proxy for the total remaining processing time.
            #    The increase in makespan due to this assignment is roughly:
            #    max(current_makespan, completion_t) + (job_rem_work - dur)
            #    But since we are comparing relative increases, we can simplify.
            #    
            #    A simpler and effective proxy: 
            #    Score = completion_t + (job_rem_work - dur)
            #    This favors operations that finish early AND jobs that have little remaining work.
            #    However, we want to minimize the *increase* in the critical path.
            #    
            #    Let's use: Score = completion_t + 0.5 * job_rem_work
            #    This balances immediate completion with job urgency.
            
            score = completion_t + 0.5 * job_rem_work
            
            if score < min_critical_path_increase:
                min_critical_path_increase = score
                best_m_for_op = m
                min_completion_for_op = completion_t

        if best_m_for_op is None:
            continue
            
        # Simulate the assignment to update the global state for comparison
        simulated_free_times = dict(_machine_free_times)
        dur = op["machine_options"][best_m_for_op]
        start_t = max(time, simulated_free_times.get(best_m_for_op, 0))
        finish_t = start_t + dur
        simulated_free_times[best_m_for_op] = finish_t
        
        # Calculate the makespan estimate: max free time across all machines
        current_max_load = max(simulated_free_times.values()) if simulated_free_times else time
        
        # Final score: Combine the critical path estimate with the global makespan
        # We prioritize minimizing the makespan, but use the critical path score as a tie-breaker
        # or a secondary objective to avoid local minima.
        
        # Let's use a weighted sum:
        # Total Score = alpha * current_max_load + beta * min_critical_path_increase
        # We want to minimize both.
        
        total_score = 1.0 * current_max_load + 0.1 * min_critical_path_increase
        
        candidates.append((total_score, min_critical_path_increase, i, best_m_for_op))
        
    if not candidates:
        return 0

    # Sort candidates by total score
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
