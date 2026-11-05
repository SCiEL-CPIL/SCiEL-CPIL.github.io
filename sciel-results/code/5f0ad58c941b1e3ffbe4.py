"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: Critical Path / Longest Remaining Path (LRP) with SPT tie-breaker.
    
    Strategy: 
    1. Estimate the total remaining processing time for each job by summing the 
       minimum possible durations of all its remaining operations. This approximates 
       the critical path length for that job.
    2. Prioritize the job with the longest estimated remaining time (most urgent 
       in terms of global makespan impact).
    3. If there are ties in critical path length (or to break ties among operations 
       from the same job, though usually only one op per job is ready), select the 
       operation with the shortest processing time (SPT) to free up machine capacity 
       quickly.
    
    This shifts focus from local machine availability (which can lead to local 
    minima or starvation) to global structural urgency, ensuring that jobs with 
    long chains of operations are started early to minimize the overall makespan.
    """
    # Deadline guard: if we are running out of time, return the first valid index
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    # For each ready operation, we need:
    # 1. The estimated critical path length of its job (sum of min durations of remaining ops)
    # 2. The processing time of the operation itself (min duration among machine options)
    
    # We assume 'remaining_ops' is a list of dictionaries similar to the current op,
    # or at least contains 'machine_options'. 
    # The problem statement says: "each ready operation is a dict with ... remaining_ops".
    # Usually remaining_ops includes the current op or the future ops. 
    # Let's assume remaining_ops is a list of future ops (excluding the current one) 
    # OR including it. 
    # Standard definition: "remaining_ops" usually means the operations in the job 
    # that have not yet been started. If the current op is ready, it is likely 
    # included in remaining_ops or remaining_ops are the *other* remaining ones.
    # Let's look at the keys: 'remaining_work' is also provided. 
    # 'remaining_work' is likely the sum of durations. 
    # However, the directive says: "estimated by summing the minimum possible 
    # durations of all remaining operations".
    # This implies we should calculate this sum ourselves if 'remaining_work' 
    # is not exactly that (e.g., if remaining_work uses specific machine assignments 
    # rather than min durations).
    
    # Let's calculate the critical path estimate for each unique job in ready_operations.
    # Since multiple operations from different jobs are ready, we evaluate the job 
    # associated with each ready operation.
    
    best_idx = 0
    best_key = (float('inf'), float('inf')) # (neg_critical_path, processing_time)
    # We want to MAXIMIZE critical path (urgency) and MINIMIZE processing time (tie-break).
    # So we can sort by (-critical_path, processing_time) and pick the first.
    # Or simply track the best tuple where lower is better for the second element 
    # and higher for the first.
    
    # To avoid recalculating the critical path for the same job multiple times 
    # (if multiple ops from the same job were ready, which shouldn't happen in 
    # a fixed-order job shop, but just in case), we could cache, but with 10 jobs 
    # it's negligible.
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        # 1. Calculate the processing time for this specific operation (min duration)
        op_duration = min(machine_options.values())
        
        # 2. Calculate the estimated critical path length for the job
        # remaining_ops contains the list of remaining operations.
        # We assume the current operation is part of the job's sequence.
        # The "remaining operations" for the critical path estimate should include 
        # the current operation and all subsequent ones.
        # The key 'remaining_ops' is provided. Let's assume it lists all ops 
        # not yet completed.
        
        remaining_ops = op.get("remaining_ops", [])
        
        # If remaining_ops is empty, use the current op's duration
        if not remaining_ops:
            crit_path_est = op_duration
        else:
            # Sum the minimum duration of each remaining operation
            # Note: 'remaining_ops' might be a list of dicts with 'machine_options'
            total_min_dur = 0.0
            for r_op in remaining_ops:
                r_m_opts = r_op.get("machine_options", {})
                if r_m_opts:
                    total_min_dur += min(r_m_opts.values())
                else:
                    # Fallback if structure is different
                    total_min_dur += 1.0
            
            # Ensure the current operation is included. 
            # Usually 'remaining_ops' in these interfaces includes the current op 
            # if it's the next one to be scheduled. If it excludes it, we add it.
            # Given the ambiguity, summing the min durations of 'remaining_ops' 
            # is the standard interpretation of "summing the minimum possible 
            # durations of all remaining operations".
            # If 'remaining_ops' excludes the current one, we should add op_duration.
            # Let's check typical patterns. Often 'remaining_ops' is the list 
            # of operations in the job that are NOT yet started. Since the current 
            # op is ready, it is NOT started. So it should be in the list.
            # If it's not, the estimate is slightly low, but relative ordering 
            # between jobs should still be roughly preserved.
            
            crit_path_est = total_min_dur
            
            # Heuristic safeguard: If the sum of remaining_ops seems to exclude 
            # the current op (e.g., if the sum is significantly less than 
            # remaining_work which might include it), it's hard to tell without 
            # ground truth. We will assume 'remaining_ops' contains all pending ops.
        
        # We want to select the job with the LONGEST critical path.
        # And among those, the one with the SHORTEST current operation duration.
        # So we minimize (-crit_path_est, op_duration)
        
        current_key = (-crit_path_est, op_duration)
        
        if current_key < best_key:
            best_key = current_key
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
