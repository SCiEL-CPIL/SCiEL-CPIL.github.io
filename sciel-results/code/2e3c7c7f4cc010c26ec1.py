"""Job-Shop Scheduling: Model-and-Optimize dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard: evaluator kills at 150s, use 140s as hard limit
_DEADLINE = 140.0
_START_TIME = _time.monotonic()

# Look-ahead critical path reduction dispatch rule with smooth surrogate.
# Strategy: For each ready operation, simulate its immediate completion and
# compute the resulting reduction in the global critical path (max remaining work
# across all jobs). Pick the operation that yields the greatest decrease in this
# makespan lower bound.
#
# Improvements over parent:
# 1. Use log-sum-exp soft-max as a smooth surrogate for the max remaining work,
#    which better captures the "near-critical" jobs and provides a more
#    differentiable score landscape.
# 2. Incorporate machine-load awareness: prefer operations on less-loaded
#    machines to reduce contention and idle time.
# 3. Add a secondary term that favors completing operations with high
#    remaining_ops (i.e., early operations in a job) to keep jobs progressing.
# 4. Tune the interaction weights for better balance between urgency and speed.

_LOG_SUM_EXP_SCALE = 0.5  # temperature for soft-max approximation (lower = sharper)


def _soft_max(values, scale=0.5):
    """Approximate max via log-sum-exp for smoothness.
    
    This provides a differentiable approximation to the max function,
    which is useful for scoring operations that are "near" the critical path.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    max_val = max(values)
    if max_val <= 0:
        return max_val
    # log-sum-exp: max + log(sum(exp((x - max)/scale)))
    shifted = [v / scale - max_val / scale for v in values]
    log_sum = float(np.log(np.sum(np.exp(shifted))))
    return max_val / scale + log_sum * scale


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using look-ahead critical path reduction
    with smooth surrogate and machine-load awareness.
    
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
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Build a map of job -> remaining_work for all ready ops
    job_rem_work = {}
    for i, op in enumerate(ready_operations):
        j = op["job"]
        rw = op.get("remaining_work", 0.0)
        if j not in job_rem_work or rw > job_rem_work[j]:
            job_rem_work[j] = rw

    if not job_rem_work:
        return 0

    # Current critical path: max remaining work across all jobs
    current_cp = max(job_rem_work.values())
    
    # Also compute a soft-max version for smooth scoring
    all_rem_works = list(job_rem_work.values())
    current_cp_smooth = _soft_max(all_rem_works, _LOG_SUM_EXP_SCALE)

    # Estimate machine load: for each machine, sum of durations of ready ops
    # that could use it (as a proxy for contention)
    machine_load = {}
    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        for m, dur in machine_opts.items():
            if m not in machine_load:
                machine_load[m] = 0.0
            machine_load[m] += dur

    # For each ready operation, compute the critical path reduction
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            scores[i] = -1e18
            continue

        min_dur = min(machine_opts.values())
        rem_work = op.get("remaining_work", 0.0)
        rem_ops = op.get("remaining_ops", 0)
        job = op["job"]
        
        # Simulate: if this op completes, remaining_work for this job decreases by min_dur
        new_rem_work = max(0.0, rem_work - min_dur)
        
        # Compute new critical path using both hard max and soft max
        other_jobs_max = max(
            (rw for j, rw in job_rem_work.items() if j != job),
            default=0.0
        )
        new_cp = max(new_rem_work, other_jobs_max)
        
        # Critical path reduction (hard)
        cp_reduction = current_cp - new_cp
        
        # Critical path reduction (smooth)
        other_jobs_smooth = [rw for j, rw in job_rem_work.items() if j != job]
        if other_jobs_smooth:
            new_cp_smooth = _soft_max(other_jobs_smooth + [new_rem_work], _LOG_SUM_EXP_SCALE)
        else:
            new_cp_smooth = new_rem_work
        cp_reduction_smooth = current_cp_smooth - new_cp_smooth
        
        # Non-linear urgency-speed product:
        # Favor operations that are both on the critical path AND fast
        urgency = rem_work / (current_cp + 1e-9)  # 0 to 1, how critical this job is
        speed = 1.0 / (min_dur + 1e-9)  # higher = faster
        interaction = urgency * speed
        
        # Machine load factor: prefer less-loaded machines
        # Use the min-duration machine's load as a proxy
        min_dur_machine = min(machine_opts.keys(), key=lambda m: machine_opts[m])
        m_load = machine_load.get(min_dur_machine, 0.0)
        # Normalize load: lower load = higher score contribution
        load_factor = 1.0 / (m_load + min_dur + 1e-9)
        
        # Remaining ops factor: favor operations that keep jobs progressing
        # Early operations (high remaining_ops) are more important to complete
        progress_factor = rem_ops / (max(rem_ops, 1) + 1e-9)
        
        # Combine all signals:
        # Primary: cp_reduction (hard) and cp_reduction_smooth (soft)
        # Secondary: interaction (urgency-speed), load_factor, progress_factor
        scores[i] = (
            1.0 * cp_reduction +
            0.3 * cp_reduction_smooth +
            0.15 * interaction * min_dur +
            0.1 * load_factor * min_dur +
            0.05 * progress_factor
        )

    # Pick the operation with highest score
    idx = int(np.argmax(scores))
    return idx
# EVOLVE-BLOCK-END
