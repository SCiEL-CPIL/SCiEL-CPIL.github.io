"""Job-shop dispatch rule: Critical-path greedy with WSPT smoothing and contention avoidance."""
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
    
    Strategy: Greedy critical-path approximation with WSPT smoothing.
    1. Primary signal: Critical path urgency. Jobs with more remaining work (proxy for
       critical path length) should be prioritized to minimize makespan.
    2. Secondary signal: WSPT (Weighted Shortest Processing Time) to break ties and
       prioritize efficiency (shorter processing times).
    3. Contention smoothing: Use soft-min over machine options for a smoother estimate
       of effective processing time, and apply a mild contention penalty for machines
       with many competing operations.
    4. Select the operation with the highest combined score.
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
    
    # Calculate machine contention
    # A machine is "contended" if many operations have it as their shortest-duration option
    machine_counts = {}
    for m in best_machines:
        if m is not None:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    # Normalize remaining_work for critical path urgency
    max_rw = max(remaining_works) if remaining_works else 1.0
    if max_rw <= 0:
        max_rw = 1.0
    
    # Calculate scores
    scores = []
    for i in range(n):
        rw = remaining_works[i]
        min_d = min_durations[i]
        best_m = best_machines[i]
        
        eps = 1e-6
        
        # Critical path urgency: normalized remaining work
        # This is the primary signal for minimizing makespan
        cp_urgency = rw / max_rw
        
        # WSPT score: remaining_work / processing_time
        # This rewards efficiency (shorter ops) and urgency (more remaining work)
        wspt_score = rw / (min_d + eps)
        
        # Normalize WSPT for combination
        # Use soft-min for smoother effective processing time estimate
        mo = ready_operations[i].get("machine_options", {})
        if mo:
            durations = list(mo.values())
            soft_min_d = _soft_min(durations, beta=5.0)
            wspt_smooth = rw / (soft_min_d + eps)
        else:
            wspt_smooth = wspt_score
        
        # Contention penalty: mild penalty for heavily contended machines
        count = machine_counts.get(best_m, 1) if best_m is not None else 1
        contention_factor = 1.0 / (1.0 + 0.5 * (count - 1))
        
        # Combine: primary weight on critical path urgency, secondary on WSPT
        # Higher cp_urgency means the job is more critical for makespan
        # WSPT provides efficiency tie-breaking
        # Blend: 0.6 * cp_urgency + 0.4 * normalized_wspt
        # Normalize wspt_smooth to [0, 1] range for combination
        max_wspt = max(remaining_works[j] / (min_durations[j] + eps) for j in range(n)) if n > 0 else 1.0
        if max_wspt <= 0:
            max_wspt = 1.0
        
        wspt_normalized = wspt_smooth / max_wspt
        
        # Final score: weighted combination
        score = (0.6 * cp_urgency + 0.4 * wspt_normalized) * contention_factor
        
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
