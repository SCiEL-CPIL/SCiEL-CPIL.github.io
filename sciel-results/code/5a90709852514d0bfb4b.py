"""Job-Shop Scheduling: Critical-path reduction dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock.
_DEADLINE = 145.0
_START_TIME = _time.monotonic()


def _check_deadline():
    return _time.monotonic() - _START_TIME > _DEADLINE


def _fast_dispatch(ready_operations, time):
    """Fast fallback: shortest processing time first (SPT) on min machine option."""
    if not ready_operations:
        return 0
    durations = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durations.append(min(mo.values()))
        else:
            durations.append(0.0)
    return int(np.argmin(durations))


def dispatch(ready_operations, time):
    """
    Dispatch rule that simulates immediate completion of each ready operation
    and picks the one that most reduces the global critical path (max remaining work
    across all jobs), with tie-breaking by shortest processing time and then by
    number of remaining ops.
    """
    if not ready_operations:
        return 0
    if _check_deadline():
        return _fast_dispatch(ready_operations, time)

    n = len(ready_operations)

    # Build a map: job -> current remaining_work
    job_remaining_work = {}
    for op in ready_operations:
        job_id = op["job"]
        if job_id not in job_remaining_work:
            job_remaining_work[job_id] = op.get("remaining_work", 0.0)

    # Current critical path = max remaining work across all jobs that have ready ops
    # But we should consider ALL jobs, not just those with ready ops.
    # However, we only have data for jobs with ready ops. The global critical path
    # includes jobs not currently ready, but we can't see them. So we approximate
    # using the jobs we can see.
    current_critical = max(job_remaining_work.values()) if job_remaining_work else 0.0

    # For each ready operation, simulate its completion and compute new critical path
    scores = np.zeros(n)
    proc_times = np.zeros(n)

    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_dur = min(mo.values())
        else:
            min_dur = 0.0
        proc_times[i] = min_dur

        job_id = op["job"]
        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)

        # After completing this operation, the job's remaining work decreases by min_dur
        # (assuming we pick the best machine option)
        new_job_remaining = remaining_work - min_dur

        # Update the critical path: the max of all other jobs' remaining work and this job's new remaining
        # We need to know the max remaining work among all OTHER jobs
        # Since we only have info on jobs with ready ops, we approximate:
        # - If this job was the unique max, the new critical is max(second_max, new_job_remaining)
        # - Otherwise, the critical path stays the same (unless new_job_remaining > current_critical, which it won't)

        # Find the second max remaining work among all jobs in job_remaining_work
        # We can do this efficiently by tracking top two values
        if len(job_remaining_work) <= 1:
            new_critical = new_job_remaining
        else:
            # Find max and second max
            sorted_works = sorted(job_remaining_work.values(), reverse=True)
            max_work = sorted_works[0]
            second_max = sorted_works[1] if len(sorted_works) > 1 else 0.0

            if job_remaining_work[job_id] == max_work:
                # This job was the max. New critical is max(second_max, new_job_remaining)
                new_critical = max(second_max, new_job_remaining)
            else:
                # This job was not the max, so critical path unchanged
                new_critical = max_work

        # Reduction in critical path
        reduction = current_critical - new_critical
        scores[i] = reduction

    # If all reductions are zero (e.g., no job is the critical path), fall back to a
    # composite heuristic: prefer short processing time and jobs closer to completion
    if np.allclose(scores, 0.0):
        for i, op in enumerate(ready_operations):
            mo = op["machine_options"]
            min_dur = min(mo.values()) if mo else 0.0
            remaining_ops = op.get("remaining_ops", 0)
            remaining_work = op.get("remaining_work", 0.0)
            # Prefer short proc time, then fewer remaining ops, then less remaining work
            scores[i] = -min_dur * 1.0
            if remaining_ops > 0:
                scores[i] -= 0.1 / remaining_ops
            scores[i] -= 0.01 * remaining_work

    # Pick the operation with the highest score (greatest reduction)
    best_idx = int(np.argmax(scores))

    # Tie-breaking: if multiple have same score, pick shortest proc time, then fewer remaining ops
    max_score = scores[best_idx]
    tied = [i for i in range(n) if abs(scores[i] - max_score) < 1e-9]
    if len(tied) > 1:
        # Sort tied by (proc_time, remaining_ops) ascending
        tied_info = []
        for i in tied:
            mo = ready_operations[i]["machine_options"]
            min_dur = min(mo.values()) if mo else 0.0
            rem_ops = ready_operations[i].get("remaining_ops", 0)
            tied_info.append((min_dur, rem_ops, i))
        tied_info.sort()
        best_idx = tied_info[0][2]

    return best_idx
# EVOLVE-BLOCK-END
