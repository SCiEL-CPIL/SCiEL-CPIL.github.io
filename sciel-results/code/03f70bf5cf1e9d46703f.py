import math
import time

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 1.0  # 1 second safety margin for the entire function call
    
    # Calculate total remaining work to determine phase
    total_remaining_work = 0
    for op in ready_operations:
        total_remaining_work += op.get("remaining_work", 0)
    
    # Heuristic for total horizon: 
    # If we don't know the total horizon, we can estimate it from the current time 
    # and the density of work. 
    # A simple proxy: if time is small, we are early. If time is large relative to 
    # the average remaining work per job, we are late.
    # Let's use a normalized time factor.
    # If total_remaining_work is 0, we are done.
    if total_remaining_work == 0:
        return 0

    # Estimate a "progress" metric. 
    # We don't know the total makespan, but we can assume that as 'time' increases, 
    # we should shift from LRPT to SPT.
    # A common heuristic is to use the ratio of current time to an estimated total time.
    # Since we don't have total time, we can use the average remaining work as a scale.
    # If time is much larger than the average remaining work of a single op, we are likely late.
    
    avg_remaining_work = total_remaining_work / len(ready_operations)
    
    # Phase factor: 0.0 is early (LRPT), 1.0 is late (SPT)
    # We map the current time to a 0-1 range.
    # If time is 0, phase = 0.
    # If time is > 3 * avg_remaining_work, phase = 1.
    # This is a rough heuristic.
    if avg_remaining_work > 0:
        phase_factor = min(1.0, max(0.0, time / (3.0 * avg_remaining_work)))
    else:
        phase_factor = 1.0

    best_score = -1
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        if not durations:
            continue
        
        # Machine Selection:
        # In the early phase, we might want to balance load, but without load info, 
        # we stick to a soft-min that becomes a hard-min as phase_factor increases.
        # Temperature decreases as phase_factor increases.
        T = 1.0 + (1.0 - phase_factor) * 2.0  # T ranges from 3.0 (early) to 1.0 (late)
        
        exp_weights = [math.exp(-d / T) for d in durations]
        total_weight = sum(exp_weights)
        
        if total_weight < 1e-15:
            min_idx = durations.index(min(durations))
            best_m_for_op = machines[min_idx]
            min_duration = durations[min_idx]
        else:
            # Expected duration under soft-min
            exp_duration = sum(d * w for d, w in zip(durations, exp_weights)) / total_weight
            # But for the score, we should probably use the min duration or a weighted one.
            # Let's use the min duration for the "urgency" calculation but acknowledge the softness.
            # Actually, for the composite score, using the min duration is safer for urgency.
            min_idx = durations.index(min(durations))
            best_m_for_op = machines[min_idx]
            min_duration = durations[min_idx]
            
            # If we are in the late phase, we strictly want the min duration.
            # If early, maybe we don't care as much about the specific machine duration 
            # as long as the job is urgent.
            
        if min_duration <= 0:
            min_duration = 1e-9

        # Operation Scoring:
        # Blend of LRPT and SPT based on phase_factor.
        # LRPT Score: Higher is better (larger remaining work).
        # SPT Score: Higher is better (shorter current processing time).
        
        lrpt_score = remaining_work
        spt_score = 1.0 / (min_duration + 1e-9)
        
        # Normalize scores to make them comparable?
        # remaining_work can be large, spt_score can be large.
        # Let's use a weighted sum.
        
        # Weight for LRPT: (1 - phase_factor)
        # Weight for SPT: phase_factor
        
        # Also incorporate urgency: remaining_work / min_duration
        urgency = remaining_work / (min_duration + 1e-9)
        
        # Composite Score:
        # Early: Focus on LRPT (start big jobs)
        # Late: Focus on SPT (finish small jobs)
        
        score = (1.0 - phase_factor) * lrpt_score + phase_factor * spt_score
        
        # Add a small urgency bonus to break ties or encourage progress
        score += 0.1 * urgency
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op
            
            # Check deadline
            if time.time() > deadline:
                break

    if best_machine is None:
        return 0

    return (best_index, best_machine)
