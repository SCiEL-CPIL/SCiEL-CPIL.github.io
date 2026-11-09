"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Deadline guard variables
_START_TIME = None
_DEADLINE = 110.0  # Safety margin before 120s limit

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time_module.time()

def dispatch(ready_operations, time):
    """
    Dispatches the next ready operation using critical-path-aware urgency.
    Prioritizes jobs whose remaining critical path length (sum of minimum 
    processing times for all subsequent ops) is longest, ensuring bottleneck
    jobs are never delayed. Machine selection uses shortest available duration.
    """
    _init_deadline()
    
    if not ready_operations:
        return 0

    # If we are close to the deadline, return a simple fallback to avoid complex calculation overhead
    if time_module.time() - _START_TIME > _DEADLINE:
        return 0

    best_score = -1
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue

        # Find the machine with the shortest duration for this operation
        min_duration = float('inf')
        best_m_for_op = None
        
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                best_m_for_op = m

        if min_duration == float('inf'):
            continue

        # Critical-path urgency: estimate remaining critical path length.
        # remaining_work includes the current op's work plus all future ops.
        # A better proxy: if we had per-op durations we'd sum min durations.
        # With only aggregate remaining_work and remaining_ops, we approximate
        # the average min-duration per remaining op and scale.
        # 
        # Primary metric: remaining_work (total work left in job)
        # Secondary: work-to-time ratio captures urgency when current op is short
        # Tertiary: remaining_ops as a tie-breaker for longer chains
        #
        # The key insight: jobs with more remaining work relative to their 
        # current processing time are more urgent because delaying them
        # bottlenecks the entire job's completion.
        
        # Work-to-time ratio: high when current op is fast relative to total work
        # This prioritizes ops where starting now gives maximum throughput
        work_time_ratio = remaining_work / (min_duration + 1.0)
        
        # Direct urgency: more remaining work means longer critical path
        # Weight this heavily since it directly reflects makespan contribution
        urgency = remaining_work
        
        # Balance: combine urgency with work-to-time ratio
        # The ratio term helps when two jobs have similar remaining work
        # but one has a much shorter current operation (less blocking)
        score = 0.6 * urgency + 0.4 * work_time_ratio
        
        # Small bonus for jobs with more remaining ops (longer precedence chain)
        score += 0.05 * remaining_ops

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Return the index and the selected machine
    # The interface allows returning (index, machine)
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
