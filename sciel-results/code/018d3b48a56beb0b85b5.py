"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track elapsed time to ensure we return quickly.
_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 145.0  # safety margin of 5s

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job identifier
            - op: operation identifier
            - machine_options: dict {machine: duration}
            - remaining_ops: number of remaining operations in this job
            - remaining_work: total remaining processing time for this job
        time: current simulation time
    
    Returns:
        index of the ready operation to start (int), 
        or (index, machine) tuple to specify machine assignment.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with minimum duration to reduce makespan
        best_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
        return (0, best_machine)
    
    # Check deadline
    if _time.monotonic() - _DEADLINE_START > _DEADLINE_LIMIT:
        # Fallback: just pick the one with most remaining work
        return int(max(range(n), key=lambda i: ready_operations[i]["remaining_work"]))
    
    # Composite heuristic for flexible job-shop:
    # 1. Prioritize jobs with more remaining work (MWL - Most Work Load)
    # 2. Among ties or as a tiebreaker, consider the minimum available machine duration
    # 3. Prefer assigning to the machine with the shortest processing time
    #
    # Key insight for FJSP: 
    # - Jobs with more remaining work should be prioritized to avoid them becoming bottlenecks
    # - Choosing the fastest machine for an operation reduces that operation's completion time
    # - We can also consider the "load" on machines, but we don't have direct access to machine states
    #   beyond what's in machine_options (durations)
    #
    # Strategy: 
    #   score = remaining_work * 1.0 + (1.0 / max(min_duration, 0.01)) * 0.1
    #   This heavily weights remaining work, with a small bonus for operations that can be done quickly.
    #
    # Return (index, best_machine) where best_machine minimizes duration.
    
    best_idx = 0
    best_score = -float('inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        machine_opts = op["machine_options"]
        
        if not machine_opts:
            # No machine options; just use remaining work
            score = rem_work
            m = None
        else:
            min_dur = min(machine_opts.values())
            # Small bonus for shorter minimum duration
            # Use 1/min_dur to reward faster operations, scaled down
            score = rem_work + 0.1 / max(min_dur, 0.01)
            m = min(machine_opts.items(), key=lambda x: x[1])[0]
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = m
    
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
