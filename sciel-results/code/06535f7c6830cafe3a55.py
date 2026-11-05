"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0  # safety margin before 120s limit


def _check_deadline():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-compute base features
    remaining_works = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    
    # We need to know the total remaining work of the job for each ready op
    # to calculate the "new" remaining work if this op completes.
    job_remaining_works = {}
    for op in ready_operations:
        job_id = op["job"]
        # remaining_work in the dict is usually the total remaining work for the job
        # including the current op. We assume this is the case based on standard JSS definitions.
        # If remaining_work is just the current op, we'd need to sum, but usually it's job-level.
        # Let's assume op["remaining_work"] is the total remaining work for the job.
        job_remaining_works[job_id] = op["remaining_work"]
        remaining_works[len(job_remaining_works) - 1] = op["remaining_work"]
        durations = list(op["machine_options"].values())
        min_durations[len(job_remaining_works) - 1] = min(durations) if durations else 0.0

    # Calculate the current global critical path lower bound
    # This is the max remaining work across all jobs that are currently "active" (have ready ops)
    # Note: Jobs that are not ready (waiting for previous op) are not in ready_operations, 
    # but their remaining work still contributes to the makespan. 
    # However, we only have access to ready_operations. 
    # A more accurate lower bound would include all jobs, but we only see ready ones.
    # We can approximate the "active" critical path by looking at the ready ops' jobs.
    
    # Actually, the makespan is determined by the job with the most remaining work.
    # If we complete op i, the remaining work of its job decreases by the duration of op i.
    # We want to minimize the *maximum* remaining work after the step.
    
    # Let's refine the "Critical Path Reduction" logic:
    # 1. Identify the current maximum remaining work among all jobs (approximated by ready ops' jobs).
    # 2. For each ready op, simulate its completion.
    #    - If the op's job was the one with the max remaining work, the max might decrease.
    #    - If it wasn't, the max stays the same (unless the op's job becomes the new max, which is unlikely if it was lower).
    # 3. We want to pick the op that reduces the "bottleneck" the most.
    
    # Better proxy: 
    # Score = (Current Max Remaining Work) - (Projected Max Remaining Work if op i completes)
    # To compute this, we need to know the remaining work of ALL jobs, not just ready ones.
    # But we only have ready_operations. 
    # We can assume that the "critical path" is dominated by the jobs currently in the ready pool or just finished.
    # Let's use a simpler, robust heuristic that mimics this:
    # Prioritize operations from jobs that currently have the HIGHEST remaining work.
    # If multiple jobs have the same highest remaining work, prioritize the one with the shortest duration (SPT) 
    # to free up the machine sooner, or the one that reduces the work the most.
    
    # Let's implement the "Simulated Completion" logic more carefully using available data.
    # We don't have the full state, so we approximate.
    # The "makespan lower bound" is max(remaining_work[j]) for all jobs j.
    # We only know remaining_work for jobs in ready_operations.
    # Let's assume the global max is among these.
    
    max_rw_current = np.max(remaining_works)
    
    # For each op i, if we complete it:
    # new_rw_i = remaining_works[i] - min_durations[i] (using min duration as proxy for actual)
    # The new global max would be max(max_rw_current, new_rw_i, max of other jobs' rw)
    # Since we don't know other jobs' rw, we assume the "other jobs" max is either max_rw_current (if not the same job) or something else.
    # This is tricky.
    
    # Alternative Dynamic Strategy: "Urgency × Blocking"
    # Score = (Remaining Work of Job) / (Min Duration of Op)
    # This rewards jobs with lots of work (urgency) and short ops (efficiency/blocking reduction).
    # This is a non-linear combination (division) rather than linear weighting.
    
    # Let's try: Score = remaining_works / (min_durations + epsilon)
    # This effectively implements "Longest Remaining Job First" but scaled by "Speed".
    
    scores = remaining_works / (min_durations + 1e-9)
    
    # Tie-breaking: if scores are close, prefer shorter duration (SPT) to reduce machine idle time
    # Or prefer fewer remaining ops?
    # Let's stick to the primary score.
    
    best_idx = int(np.argmax(scores))
    
    if _check_deadline():
        return 0

    return best_idx
# EVOLVE-BLOCK-END
