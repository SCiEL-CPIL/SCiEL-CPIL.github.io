"""Job-shop dispatch rule: refined multi-criteria with smooth contention and dynamic CR."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 135.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def _soft_min(values, beta=5.0):
    """Soft-min via log-sum-exp for smoothness."""
    if not values:
        return 0.0
    v_min = min(values)
    shifted = [v - v_min for v in values]
    log_sum = sum(math.exp(-beta * s) for s in shifted)
    if log_sum < 1e-300:
        return v_min
    return v_min - math.log(log_sum) / beta


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Critical Ratio (CR) as primary urgency signal, combined with
    smooth machine contention and remaining-work awareness.
    
    CR = (estimated_slack) / (remaining_processing_time)
    Lower CR = more urgent = higher priority.
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Precompute features for each ready operation
    remaining_work = []
    remaining_ops = []
    min_duration = []
    max_duration = []
    avg_duration = []
    est_completion = []  # estimated time to complete this op on best machine
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
            max_d = max(durations)
            avg_d = sum(durations) / len(durations)
            # Estimate completion: current time + min duration (optimistic)
            # Use a smoothed estimate via soft-min for robustness
            sm_d = _soft_min(durations, beta=4.0)
            est_comp = time + sm_d
        else:
            min_d = 0
            max_d = 0
            avg_d = 0
            est_comp = time
        
        remaining_work.append(rw)
        remaining_ops.append(ro)
        min_duration.append(min_d)
        max_duration.append(max_d)
        avg_duration.append(avg_d)
        est_completion.append(est_comp)
    
    # Normalize features to [0, 1] range for fair weighting
    def normalize(vals):
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-9:
            return [0.0] * len(vals)
        return [(v - mn) / rng for v in vals]
    
    # Compute Critical Ratio (CR) for each operation
    # CR = (estimated_slack) / (remaining_processing_time)
    # Slack estimate: we don't have true due dates, so we use remaining_work as a proxy
    # for the "deadline pressure". A job with high remaining_work relative to its
    # current processing time is under more pressure.
    # 
    # Alternative: CR = (remaining_work - elapsed_time_estimate) / remaining_work
    # where elapsed_time_estimate accounts for how long this job has been in the system.
    # Since we don't track job start times, we use a simpler formulation:
    # CR_i = (sum of remaining durations for job) / (current time + op duration)
    # Lower CR means the job has less slack relative to how long it's been running.
    
    # For each job, compute total remaining processing time
    job_remaining = {}
    for op in ready_operations:
        job_id = op.get("job", -1)
        rw = op.get("remaining_work", 0)
        if job_id not in job_remaining:
            job_remaining[job_id] = 0
        job_remaining[job_id] = max(job_remaining[job_id], rw)
    
    # Compute CR values
    cr_values = []
    for i in range(n):
        job_id = ready_operations[i].get("job", -1)
        job_rw = job_remaining.get(job_id, remaining_work[i])
        rw_safe = remaining_work[i] + 1.0
        # CR: ratio of job's total remaining work to current time progress
        # Lower CR = more urgent
        # Use: CR = job_rw / (time + min_duration) 
        # This captures: jobs with lots of remaining work that have been waiting are urgent
        cr = job_rw / (time + min_duration[i] + 1.0)
        cr_values.append(cr)
    
    norm_cr = normalize(cr_values)
    # Lower CR is better, so we negate for scoring (higher score = more urgent)
    
    # Normalize other features
    norm_rw = normalize(remaining_work)
    norm_ro = normalize(remaining_ops)
    norm_avg_d = normalize(avg_duration)
    norm_min_d = normalize(min_duration)
    
    # Compute NSPT: normalized shortest processing time weighted by remaining work
    nspt_values = []
    for i in range(n):
        rw_safe = remaining_work[i] + 1.0
        nspt = min_duration[i] / rw_safe
        nspt_values.append(nspt)
    
    norm_nspt = normalize(nspt_values)
    
    # Compute machine contention
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    # Adaptive weights: CR is the primary signal
    w_cr = 0.45
    w_rw = 0.15
    w_ro = 0.05
    w_nspt = 0.20
    w_avg_d = 0.05
    w_machine = 0.10
    
    # Adjust based on number of ready ops
    if n <= 3:
        w_cr = 0.50
        w_rw = 0.15
        w_ro = 0.05
        w_nspt = 0.15
        w_avg_d = 0.05
        w_machine = 0.10
    elif n >= 8:
        w_cr = 0.40
        w_rw = 0.15
        w_ro = 0.05
        w_nspt = 0.25
        w_avg_d = 0.05
        w_machine = 0.10
    
    # Compute scores (higher is better for our selection)
    scores = []
    for i in range(n):
        # CR: lower normalized CR = more urgent = higher score
        s_cr = 1.0 - norm_cr[i]
        # More remaining work = higher priority
        s_rw = norm_rw[i]
        # More remaining ops = higher priority  
        s_ro = norm_ro[i]
        # NSPT: lower normalized value = shorter duration relative to remaining work = better
        s_nspt = 1.0 - norm_nspt[i]
        s_avg_d = 1.0 - norm_avg_d[i]
        
        # Machine contention
        mo = ready_operations[i].get("machine_options", {})
        if mo:
            best_machine = min(mo, key=lambda m: mo[m])
            contention = machine_counts.get(best_machine, 1)
            max_contention = max(machine_counts.values()) if machine_counts else 1
            s_contention = 1.0 - (contention - 1) / max(1, max_contention - 1)
            s_machine = 0.5 * (1.0 - norm_min_d[i]) + 0.5 * s_contention
        else:
            s_machine = 0.5
        
        score = (w_cr * s_cr + 
                 w_rw * s_rw + 
                 w_ro * s_ro + 
                 w_nspt * s_nspt + 
                 w_avg_d * s_avg_d +
                 w_machine * s_machine)
        scores.append(score)
    
    # Select best
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score + 1e-9:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)

# EVOLVE-BLOCK-END
