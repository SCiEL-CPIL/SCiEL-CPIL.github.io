"""Job-shop dispatch rule: Critical Ratio (CR) heuristic with dynamic urgency and contention avoidance."""
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
    
    Strategy: Critical Ratio (CR) heuristic with dynamic urgency.
    1. Critical Ratio: slack / remaining_processing_time
       - Slack = estimated time until job completion deadline - current time
       - We estimate remaining processing time as remaining_work (sum of min durations)
       - Lower CR = more urgent (less slack relative to work remaining)
    2. Machine contention penalty: penalize operations targeting heavily contended machines
    3. WSPT tie-breaker: for similar CR values, prefer shorter processing times
    
    CR < 1 means the job is behind schedule (negative slack), CR > 1 means ahead of schedule.
    We prioritize operations with the lowest CR (most urgent).
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
    remaining_ops_list = []
    
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
        remaining_ops_list.append(ro)
        
    # Calculate machine contention
    # A machine is "contended" if many operations have it as their shortest-duration option
    machine_counts = {}
    for m in best_machines:
        if m is not None:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    # Estimate a rough makespan horizon for CR calculation
    # Use max remaining_work as a proxy for the longest remaining job path
    max_remaining_work = max(remaining_works) if remaining_works else 1.0
    if max_remaining_work <= 0:
        max_remaining_work = 1.0
    
    # Calculate scores - we want to MINIMIZE the adjusted CR
    scores = []
    for i in range(n):
        rw = remaining_works[i]
        min_d = min_durations[i]
        best_m = best_machines[i]
        ro = remaining_ops_list[i]
        
        eps = 1e-6
        
        # Critical Ratio: slack / remaining_processing
        # Slack estimation: we don't know exact deadlines, so use a heuristic
        # Assume each job has a "target completion" proportional to its total work
        # Slack = (estimated_target_time - current_time)
        # Since we don't know target times, use remaining_work as urgency proxy
        # CR = remaining_work / (remaining_ops * avg_duration_estimate)
        # Lower CR means more operations remaining with less work per op = less urgent
        # Actually, let's think differently:
        
        # CR = slack / processing_time
        # If we assume all jobs should finish by some horizon H:
        # slack_i = H - time - (estimated_remaining_time_for_job_i)
        # But we don't know H. Instead, use relative urgency:
        
        # Alternative: Use WSPT-like urgency but inverted
        # Operations with more remaining work relative to their processing speed are more urgent
        # urgency = remaining_work / (min_duration + eps)
        # This is actually WSPT, which we want to combine with CR concept
        
        # Let's compute a CR-like metric:
        # remaining_fraction = rw / max_remaining_work (how much work left relative to max)
        # ops_fraction = ro / max_ro (how many ops left)
        # A job with high remaining_fraction but low ops_fraction has large remaining ops = urgent
        
        max_ro = max(remaining_ops_list) if remaining_ops_list else 1
        if max_ro <= 0:
            max_ro = 1
            
        remaining_frac = rw / max_remaining_work
        ops_frac = ro / max_ro
        
        # CR-like: jobs with more remaining work fraction relative to ops are more urgent
        # cr_urgency = remaining_frac / (ops_frac + eps)
        # Higher cr_urgency = more urgent
        cr_urgency = remaining_frac / (ops_frac + eps)
        
        # WSPT component: rw / min_d (higher = more urgent per unit time)
        wspt_component = rw / (min_d + eps)
        
        # Normalize both to [0, 1] range for combination
        # We'll compute raw scores first, then normalize
        
        # Contention factor
        if best_m is not None:
            count = machine_counts.get(best_m, 1)
            contention_factor = 1.0 / (1.0 + count)
        else:
            contention_factor = 1.0
        
        # Combine: primary is CR urgency, secondary is WSPT, penalized by contention
        # Use a weighted combination where CR dominates
        raw_score = 0.6 * cr_urgency + 0.4 * (wspt_component / (max_remaining_work / (min_d + eps) + eps))
        score = raw_score * contention_factor
        
        scores.append(score)
        
    # Select best index (highest score)
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
            
    return int(best_idx)

# EVOLVE-BLOCK-END
