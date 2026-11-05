"""Job-shop dispatch rule: WSPT-enhanced with critical-path urgency and smooth contention."""
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
    
    Strategy: WSPT enhanced with critical-path urgency and smooth contention penalty.
    1. WSPT score: remaining_work / (min_duration + epsilon).
       - Prioritizes jobs with more remaining work (urgency).
       - Prioritizes operations with shorter processing times (efficiency).
    2. Critical-path urgency: estimated by summing minimum possible durations of all
       remaining operations in the job. Jobs with longer critical paths get higher priority.
    3. Smooth contention penalty: use soft-min over machine options to estimate the
       "effective" processing time, then penalize operations targeting heavily contended
       machines based on how many ready ops prefer that machine.
    4. Select the operation with the highest adjusted score.
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Precompute features
    min_durations = []
    best_machines = []
    remaining_works = []
    remaining_ops = []
    # Critical path estimate: sum of min durations of all remaining ops in the job
    critical_path_estimates = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            # Find machine with shortest duration
            best_m, min_d = min(mo.items(), key=lambda item: item[1])
        else:
            best_m = None
            min_d = 0
            
        min_durations.append(min_d)
        best_machines.append(best_m)
        remaining_works.append(rw)
        remaining_ops.append(ro)
        
        # Estimate critical path: use remaining_work as a proxy for total remaining processing
        # This is a structural urgency measure: jobs with more remaining work are more critical
        critical_path_estimates.append(rw)
        
    # Calculate machine contention
    # A machine is "contended" if many operations have it as their shortest-duration option
    machine_counts = {}
    for m in best_machines:
        if m is not None:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    # Calculate scores
    scores = []
    for i in range(n):
        rw = remaining_works[i]
        min_d = min_durations[i]
        best_m = best_machines[i]
        cp_est = critical_path_estimates[i]
        
        # WSPT: Weight / Processing Time
        # Use epsilon to avoid division by zero
        eps = 1e-6
        wspt_score = rw / (min_d + eps)
        
        # Critical path urgency: normalize by max critical path estimate
        max_cp = max(critical_path_estimates) if critical_path_estimates else 1.0
        if max_cp > 0:
            cp_urgency = cp_est / max_cp
        else:
            cp_urgency = 1.0
        
        # Smooth contention penalty using soft-min over machine options
        # This gives a smoother estimate of the "effective" processing time
        mo = ready_operations[i].get("machine_options", {})
        if mo:
            durations = list(mo.values())
            soft_min_d = _soft_min(durations, beta=5.0)
            # Contention factor: penalize based on how many ops target this machine
            count = machine_counts.get(best_m, 1)
            # Soft penalty: 1 / (1 + count)
            contention_factor = 1.0 / (1.0 + count)
            # Combine soft-min duration with contention
            # Use soft_min_d in WSPT for smoother behavior
            wspt_smooth = rw / (soft_min_d + eps)
            score = wspt_smooth * contention_factor * (0.7 + 0.3 * cp_urgency)
        else:
            score = wspt_score * cp_urgency
        
        scores.append(score)
        
    # Select best index
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
            
    return int(best_idx)

# EVOLVE-BLOCK-END
