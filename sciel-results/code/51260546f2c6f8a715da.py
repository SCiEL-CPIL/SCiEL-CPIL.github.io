"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin

def dispatch(ready_operations, time):
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback: pick first available or random if too late
        return 0 if ready_operations else 0

    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        # Return index and best machine
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)
    
    best_score = float('inf')
    best_idx = 0
    best_machine = None
    
    # Lookahead Heuristic:
    # For each candidate (i, machine m):
    # 1. Estimate when this machine will be free after this op: t_end_machine = time + duration
    # 2. Estimate when this job will be done: t_end_job = time + remaining_work
    #    (Note: remaining_work includes current op + future ops. 
    #     A better proxy for "job criticality" is remaining_work.
    #     A better proxy for "machine load" is duration.
    #     However, simple addition ignores that other ops might be queued.
    #     Let's refine: We want to minimize the maximum of (machine_free_time, job_completion_time)
    #     Actually, makespan is driven by the latest finishing job.
    #     So we prioritize operations that are on the "critical path".
    #     Heuristic: Score = max(time + duration, time + remaining_work)
    #     But remaining_work is usually >= duration. So max is always time + remaining_work.
    #     This would just pick the job with the most remaining work (WSPT-like).
    #     
    #     Let's try a different lookahead:
    #     If we pick op i on machine m, the machine is busy until time + d_i_m.
    #     The job i will still have remaining_work - d_i_m work left (approx, if d is part of remaining_work).
    #     Actually, remaining_work usually sums all future durations.
    #     Let's assume remaining_work includes the current operation's duration.
    #     Then, after picking op i on machine m:
    #     - Machine m is unavailable until T_m = time + d_i_m
    #     - Job i has remaining_work' = remaining_work - d_i_m
    #     
    #     We want to avoid creating a bottleneck.
    #     A common dispatching rule is "Least Load" or "Earliest Completion".
    #     Let's try minimizing the estimated completion time of the job, but penalizing machine congestion.
    #     
    #     New Strategy: "Weighted Shortest Processing Time with Machine Awareness"
    #     Score = (time + duration) / (remaining_work)  -> Higher ratio means machine is more "expensive" relative to job progress.
    #     We want to MINIMIZE this cost.
    #     Or: Score = time + duration - lambda * remaining_work
    #     
    #     Let's go with a "Critical Path" approximation:
    #     Priority = remaining_work (jobs with more work left are more critical)
    #     But we also want to use machines efficiently.
    #     
    #     Let's try: Minimize (time + duration) for the machine, but only among jobs with high remaining_work.
    #     This is complex.
    #     
    #     Let's stick to a robust structural change: **Simulated Annealing-like local search is too slow.**
    #     Let's use **Lookahead of 1 step on Machine Availability**.
    #     
    #     For each op i and machine m:
    #     cost = (time + duration[i][m])  # When machine is free
    #     benefit = remaining_work[i]     # How much work we clear
    #     We want high benefit / low cost.
    #     Ratio = remaining_work[i] / (time + duration[i][m])
    #     Maximize this ratio.
    
    for i, op in enumerate(ready_operations):
        mw = op["remaining_work"]
        if mw <= 0:
            continue
            
        for m, d in op["machine_options"].items():
            # Estimated completion time for this specific operation on this machine
            t_end = time + d
            
            # Heuristic: Maximize (Remaining Work / Estimated Completion Time)
            # This favors operations that clear a lot of work early relative to the clock time.
            # It also implicitly favors shorter durations (lower t_end).
            # It also favors jobs with lots of remaining work (higher mw).
            
            if t_end <= 0:
                continue
                
            score = mw / t_end
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = m
                
    # Fallback if no valid score found
    if best_machine is None:
        best_idx = 0
        op = ready_operations[0]
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
