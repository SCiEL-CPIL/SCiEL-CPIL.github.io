"""Job-Shop Scheduling: Combined dispatch rule with machine selection and composite scoring."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard: evaluator kills at 150s, use 140s as hard limit
_DEADLINE = 140.0
_START_TIME = _time.monotonic()

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and which machine to use.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation index
            - machine_options: {machine: duration}
            - remaining_ops: number of ops remaining in this job
            - remaining_work: total remaining processing time for this job
        time: current simulation time
    
    Returns:
        index of the ready operation to start
    """
    # Deadline guard: if we're close to the limit, just return first op
    elapsed = _time.monotonic() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-allocate arrays for performance
    min_durs = np.empty(n)
    rem_works = np.empty(n)
    rem_ops = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            min_durs[i] = np.inf
            rem_works[i] = 0.0
            rem_ops[i] = 0
            continue
        
        # Find min duration
        min_d = min(machine_opts.values())
        
        min_durs[i] = min_d
        rem_works[i] = op.get("remaining_work", 0.0)
        rem_ops[i] = op.get("remaining_ops", 0)

    # Handle cases where no machine options were valid
    valid_mask = min_durs < np.inf
    if not np.any(valid_mask):
        return 0

    # Look-ahead critical path reduction heuristic
    # Simulate completing each ready operation and measure the reduction in
    # global makespan lower bound (max remaining work across all jobs).
    # Select the operation that yields the greatest decrease.
    
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        return 0
    
    if n_valid == 1:
        return int(valid_indices[0])
    
    # Build job -> remaining_work mapping from ready operations
    # For look-ahead: if we complete operation i, job j's remaining_work decreases by min_dur_i
    # The global makespan lower bound is max(remaining_work across all jobs)
    # We want to minimize this max after completion
    
    # Get unique jobs and their current remaining work
    job_rem_work = {}
    for i, op in enumerate(ready_operations):
        j = op["job"]
        rw_val = op.get("remaining_work", 0.0)
        # Keep the max if multiple ops from same job are ready (shouldn't happen in fixed order)
        if j not in job_rem_work or rw_val > job_rem_work[j]:
            job_rem_work[j] = rw_val
    
    current_max_rw = max(job_rem_work.values()) if job_rem_work else 0.0
    
    # For each valid operation, compute the reduction in makespan lower bound
    scores = np.zeros(n_valid)
    
    for k, idx in enumerate(valid_indices):
        op = ready_operations[idx]
        job = op["job"]
        min_d = min_durs[idx]
        
        # After completing this operation, this job's remaining work decreases by min_d
        new_job_rw = job_rem_work[job] - min_d
        if new_job_rw < 0:
            new_job_rw = 0.0
        
        # New makespan lower bound: max of all other jobs' remaining work and new_job_rw
        # We need to find the max excluding this job, then include new_job_rw
        
        # Optimization: if current_max_rw > job_rem_work[job], then the max is from another job
        # and doesn't change (unless new_job_rw > that max, which it won't since new_job_rw < job_rem_work[job])
        if job_rem_work[job] < current_max_rw:
            # The max is from another job, so it stays the same
            new_max_rw = current_max_rw
        else:
            # This job had the max (or tied). Need to find second max.
            # Find the max remaining work among all other jobs
            second_max = 0.0
            for j2, rw2 in job_rem_work.items():
                if j2 != job and rw2 > second_max:
                    second_max = rw2
            new_max_rw = max(second_max, new_job_rw)
        
        # Reduction in makespan lower bound
        reduction = current_max_rw - new_max_rw
        
        # Tie-breaking: prefer shorter operations (SPT) to free machines faster
        # and fewer remaining ops to finish jobs
        spt_bonus = 0.1 / (min_d + 1e-9)
        ops_bonus = 0.02 * (5 - op.get("remaining_ops", 0))  # Normalize: 5 ops is typical
        
        scores[k] = reduction + spt_bonus + ops_bonus
    
    best_local_idx = int(np.argmax(scores))
    best_idx = int(valid_indices[best_local_idx])
    
    return best_idx
# EVOLVE-BLOCK-END
