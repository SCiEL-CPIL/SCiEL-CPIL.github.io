"""
Strategy: Greedy Critical Path with Machine Congestion Estimation.
Instead of a fixed linear combination of scores, this dispatch rule:
1. Identifies the machine option for each ready operation that minimizes 
   (duration + estimated_queue_time).
2. Estimates queue time using a proxy for machine load (inverse of duration 
   is not enough; we use a heuristic that penalizes short jobs on busy machines 
   if the job has long remaining work, but favors short jobs to clear queues 
   if the job is nearly done).
3. Selects the operation with the highest "Criticality", defined as the 
   remaining critical path length of the job, adjusted by the efficiency 
   of the chosen machine assignment.
"""

# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE_MARGIN = 2.0
_START_TIME = None

# Global cache for machine congestion estimation
# We don't have direct access to machine state, so we estimate "busyness" 
# by tracking how often a machine is requested for short vs long jobs in 
# recent dispatches, or simply use a heuristic based on the current batch.
# Since we only see 'ready_operations', we assume a static heuristic for 
# congestion unless we maintain state. 
# Let's maintain a simple state of machine "heat" (recently used frequency) 
# to penalize machines that are likely busy.

_machine_heat = {}

def _now():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME

def _deadline_exceeded():
    return _now() > (150.0 - _DEADLINE_MARGIN)

def _estimate_machine_load(machine_id, duration, remaining_work, remaining_ops):
    """
    Heuristic estimate of total time an operation will take on a specific machine.
    Base cost is the duration.
    We add a congestion penalty. Since we don't know the actual queue, we use
    a heuristic: if the machine is 'hot' (used frequently in recent dispatches),
    we assume it has a backlog.
    """
    heat = _machine_heat.get(machine_id, 0.0)
    # Decay heat slightly to represent time passing
    # We don't have real time steps easily accessible for decay without global clock,
    # so we just use the raw count as a proxy for congestion.
    
    # If the job is critical (high remaining work), we might prefer a machine
    # even if it's hot, but if the job is small, we prefer a cool machine.
    urgency = 1.0 / max(remaining_ops, 1)
    
    # Congestion penalty: proportional to heat and inversely proportional to urgency?
    # Actually, if a machine is hot, the wait time is high.
    # Let's assume wait time ~ heat * average_duration.
    # We don't know average duration, so we just use heat as a raw penalty.
    penalty = heat * 0.5
    
    return duration + penalty

def dispatch(ready_operations, time):
    global _machine_heat

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    if _deadline_exceeded():
        # Fallback: Longest Processing Time First (LPT) on min duration
        return int(max(range(len(ready_operations)), 
                       key=lambda i: min(ready_operations[i].get("machine_options", {}).values(), default=0)))

    n = len(ready_operations)
    best_score = -float('inf')
    best_idx = 0

    # Pre-compute job criticality (remaining work)
    # In a job shop, the critical path of a job is roughly the sum of remaining op durations.
    # 'remaining_work' is a good proxy for this.

    for i in range(n):
        op = ready_operations[i]
        opts = op.get("machine_options", {})
        if not opts:
            continue
            
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        
        # Find the best machine for this operation considering congestion
        best_machine_score = float('inf')
        best_machine_id = None
        
        for m_id, dur in opts.items():
            est_time = _estimate_machine_load(m_id, dur, remaining_work, remaining_ops)
            if est_time < best_machine_score:
                best_machine_score = est_time
                best_machine_id = m_id
        
        if best_machine_id is None:
            continue

        # Heuristic Score:
        # 1. Criticality: Jobs with more remaining work should be prioritized 
        #    to prevent them from becoming the bottleneck (Longest Remaining Path).
        # 2. Efficiency: Prefer operations that will finish quickly on their chosen machine
        #    to free up resources for others (Shortest Processing Time).
        # 3. Balance: We want to start the operation that reduces the makespan the most.
        #    Starting a long job early is good. Starting a short job early is good for flow.
        
        # Let's use a weighted sum:
        # Weight 1: Remaining Work (Critical Path) - higher is better (start it)
        # Weight 2: 1/Estimated Completion Time - higher is better (finish it fast)
        
        # Normalize-ish:
        # remaining_work is in time units.
        # 1/est_time is inverse time.
        
        # Strategy: "Longest Remaining Processing Time" (LRPT) is often strong in Job Shop.
        # It prioritizes jobs that have the most work left, assuming they are on the critical path.
        
        score_critical = remaining_work
        score_flow = 1.0 / max(best_machine_score, 0.001)
        
        # Combine: Criticality dominates, but flow helps clear queues.
        # If remaining_work is 0, it's the last op, prioritize flow.
        if remaining_work > 0:
            combined_score = 1.0 * score_critical + 0.5 * score_flow * remaining_work
        else:
            combined_score = 1000.0 + score_flow # Last ops should be very high priority to finish job

        if combined_score > best_score:
            best_score = combined_score
            best_idx = i

    # Update machine heat for the selected operation
    # This is a heuristic update. We assume the selected machine is now "busier".
    # We need to know which machine was selected.
    # We re-calculate the best machine for the selected op to update heat.
    sel_op = ready_operations[best_idx]
    sel_opts = sel_op.get("machine_options", {})
    sel_rem_work = sel_op.get("remaining_work", 0)
    sel_rem_ops = sel_op.get("remaining_ops", 1)
    
    if sel_opts:
        best_m_score = float('inf')
        best_m_id = None
        for m_id, dur in sel_opts.items():
            est_time = _estimate_machine_load(m_id, dur, sel_rem_work, sel_rem_ops)
            if est_time < best_m_score:
                best_m_score = est_time
                best_m_id = m_id
        
        if best_m_id is not None:
            _machine_heat[best_m_id] = _machine_heat.get(best_m_id, 0.0) + 1.0
            # Optional: Decay all other machines? 
            # To prevent heat from accumulating indefinitely, we could decay all.
            # But with only 5 machines, iterating is cheap.
            for k in _machine_heat:
                if k != best_m_id:
                    _machine_heat[k] *= 0.95

    return int(best_idx)
# EVOLVE-BLOCK-END
