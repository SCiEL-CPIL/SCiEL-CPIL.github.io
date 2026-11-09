"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Track start time for deadline guard
_t0 = _time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _select_machine(op, time):
    """
    Select the best machine for an operation.
    Uses a combined heuristic: prefer machines with shortest processing time,
    but also consider load balancing.
    Returns the machine key that minimizes a composite score.
    """
    machine_options = op["machine_options"]
    best_machine = None
    best_score = float('inf')
    
    for machine, duration in machine_options.items():
        # Heuristic: prefer shorter processing times
        # But also consider the work-to-time ratio to balance load
        score = duration
        best_score = score
        best_machine = machine
    
    return best_machine


def dispatch(ready_operations, time):
    """
    Dispatch rule that picks which ready operation to start next and,
    optionally, which eligible machine runs it.
    
    Strategy: 
    1. For each ready operation, compute a priority score.
    2. The priority combines:
       - Remaining work (more work = higher priority)
       - Number of remaining ops (more ops = higher priority, as it's more critical)
       - Shortest available processing time across machines (shorter = higher priority, to keep machines busy)
    3. Return (index, machine) for the highest priority operation.
    """
    # Deadline guard
    if _time.time() - _t0 > _DEADLINE:
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Precompute some values for speed
    best_scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Find the shortest processing time across all machines
        min_duration = float('inf')
        best_machine = None
        for machine, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                best_machine = machine
        
        # Composite priority score:
        # - Higher remaining work means more critical
        # - More remaining ops means this job is further from completion
        # - Shorter processing time means we can complete it faster (reduces idle time)
        
        # Weighted combination:
        # Primary: remaining work (to finish large jobs early)
        # Secondary: remaining ops (jobs with more ops left are more critical)
        # Tertiary: shortest processing time (to keep machines busy)
        
        # Normalize-ish approach: use raw values with weights
        score = remaining_work * 1.0 + remaining_ops * 0.1 + (1.0 / (min_duration + 1e-9)) * 0.01
        
        best_scores.append(score)
        best_machines.append(best_machine)
    
    # Find the index with the highest score
    best_idx = 0
    best_score = best_scores[0]
    for i in range(1, n):
        if best_scores[i] > best_score:
            best_score = best_scores[i]
            best_idx = i
    
    # Return (index, machine) to explicitly select the best machine
    return (best_idx, best_machines[best_idx])
# EVOLVE-BLOCK-END
