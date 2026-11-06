"""Job-Shop Scheduling Dispatch Rule"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Uses a look-ahead simulation that estimates the reduction in global critical path
    if each ready operation were completed immediately, accounting for machine conflicts.
    Selects the operation that yields the greatest decrease in the makespan lower bound.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    if n == 1:
        return 0
    
    # Precompute basic features
    min_durations = np.array([min(op["machine_options"].values()) if op["machine_options"] else 1.0 for op in ready_operations], dtype=np.float64)
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Compute job-level remaining work (max remaining work per job = critical path for that job)
    job_remaining_work = {}
    for op in ready_operations:
        job = op["job"]
        if job not in job_remaining_work:
            job_remaining_work[job] = op["remaining_work"]
        else:
            job_remaining_work[job] = max(job_remaining_work[job], op["remaining_work"])
    
    # Current global critical path
    current_cp = max(job_remaining_work.values()) if job_remaining_work else 0.0
    
    # Machine contention map: for each machine, track which ready ops use it
    machine_ops = {}
    for i, op in enumerate(ready_operations):
        for mach in op["machine_options"]:
            if mach not in machine_ops:
                machine_ops[mach] = []
            machine_ops[mach].append(i)
    
    # Look-ahead: for each ready operation, estimate the new CP after completing it
    # We simulate completion and see how much the global CP decreases
    cp_reductions = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        op = ready_operations[i]
        job = op["job"]
        
        # After completing op i, job's remaining work decreases
        # The job's remaining work becomes max of other ops' remaining work for that job
        # But we only see ready ops here; the actual remaining work for the job after this op
        # is determined by the next op in the job's sequence
        
        # Estimate: the reduction in this job's critical path
        # If this op was the one determining the job's max remaining work, the new max
        # would be the next-highest remaining work for that job among ready ops
        # If not, the job's CP stays the same
        
        # Find all ready ops for this job
        same_job_ops = [j for j in range(n) if ready_operations[j]["job"] == job]
        
        if len(same_job_ops) == 1:
            # Only this op is ready for this job; after completion, job's remaining work
            # drops to whatever is not yet ready (we approximate as 0 if no other ready ops)
            # But actually remaining_work includes all remaining ops, so after this op completes,
            # the job still has remaining_work - duration of this op (approximately)
            # However, remaining_work is the total remaining work including this op
            # So after completing this op, job remaining work ≈ remaining_work[i] - min_duration[i]
            # But we don't know the exact duration chosen; use min as optimistic estimate
            new_job_rw = max(0.0, remaining_works[i] - min_durations[i])
        else:
            # Multiple ops ready for same job; after completing op i,
            # the job's remaining work is max of remaining_work of other ready ops for this job
            # (this is an approximation - in reality the job's total remaining work decreases)
            other_rws = [remaining_works[j] for j in same_job_ops if j != i]
            new_job_rw = max(other_rws) if other_rws else 0.0
            # Also account for the fact that completing op i reduces total job work
            # The true remaining work after completion is remaining_work[i] - actual_duration
            # Since remaining_work[i] >= remaining_work[j] for all j in same job (it's the max),
            # and we're removing op i, the new max is max of others
            # But the actual total remaining work for the job decreases by the duration of op i
            # So a better estimate: new_job_rw = max(0, remaining_works[i] - min_durations[i], max(other_rws))
            # Actually remaining_work is per-op, representing the total remaining work for the job
            # from that op's perspective. They should all be similar if they're in the same job.
            # Let's use: new_job_rw = max(remaining_works[i] - min_durations[i], max(other_rws) if other_rws else 0)
            alt_estimate = remaining_works[i] - min_durations[i]
            new_job_rw = max(new_job_rw, alt_estimate) if alt_estimate > 0 else new_job_rw
        
        # New global CP after completing op i
        # The new CP is max of: new_job_rw for this job, and old CP for all other jobs
        other_jobs_cp = max([rw for j, rw in job_remaining_work.items() if j != job], default=0.0)
        new_cp = max(new_job_rw, other_jobs_cp)
        
        # Reduction in CP
        cp_reductions[i] = current_cp - new_cp
    
    # Machine conflict penalty: if multiple ready ops share a machine, completing one
    # may delay others. Penalize ops that would cause high contention.
    conflict_penalty = np.zeros(n, dtype=np.float64)
    for mach, ops_on_mach in machine_ops.items():
        if len(ops_on_mach) > 1:
            # Total work on this machine from ready ops
            total_mach_work = sum(min_durations[j] for j in ops_on_mach)
            for j in ops_on_mach:
                # Penalty proportional to how much this op delays others
                # If this op is long, it blocks others more
                conflict_penalty[j] += min_durations[j] * (len(ops_on_mach) - 1) / (total_mach_work + 1e-9)
    
    # Composite score: prefer ops with high CP reduction and low conflict penalty
    # Normalize CP reduction
    max_reduction = np.max(cp_reductions)
    if max_reduction > 1e-12:
        cp_norm = cp_reductions / max_reduction
    else:
        cp_norm = np.zeros(n)
    
    # Normalize conflict penalty (lower is better)
    max_penalty = np.max(conflict_penalty)
    if max_penalty > 1e-12:
        penalty_norm = conflict_penalty / max_penalty
    else:
        penalty_norm = np.zeros(n)
    
    # Final score: high CP reduction, low conflict penalty
    final_scores = cp_norm * 0.70 - penalty_norm * 0.30
    
    # Secondary tie-breakers: prefer shorter processing time, fewer remaining ops
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    spt_norm = 1.0 - normalize(min_durations)
    rem_ops_norm = 1.0 - normalize(remaining_ops)
    
    # Add small tie-breaking terms
    final_scores += spt_norm * 0.05 + rem_ops_norm * 0.05
    
    # Select best
    max_score = np.max(final_scores)
    close_to_max = np.where(final_scores > max_score - 1e-9)[0]
    if len(close_to_max) > 1:
        tied_durations = min_durations[close_to_max]
        best_tie_idx = close_to_max[np.argmin(tied_durations)]
        return int(best_tie_idx)
    
    return int(np.argmax(final_scores))
# EVOLVE-BLOCK-END
