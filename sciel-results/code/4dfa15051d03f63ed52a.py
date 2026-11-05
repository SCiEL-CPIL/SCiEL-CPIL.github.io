"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_start_time = time.time()
_DEADLINE = 120.0

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a fast heuristic
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Fast fallback: pick the op with the smallest min duration
        best_idx = 0
        best_min = float('inf')
        for i, op in enumerate(ready_operations):
            min_dur = min(op["machine_options"].values())
            if min_dur < best_min:
                best_min = min_dur
                best_idx = i
        return int(best_idx)

    # STRUCTURAL CHANGE: Machine-centric bottleneck heuristic
    # Instead of job-centric urgency (remaining work), we score each ready operation
    # by the "bottleneck reduction" it provides: how much it reduces the minimum
    # completion time achievable on the least-loaded eligible machine.
    
    # We don't have direct machine load info, so we use a proxy:
    # For each op, consider the machine that would give the shortest completion time.
    # Score = min over machines of (duration on that machine) / (number of other ready ops 
    # that could also use that machine)
    # This approximates: if a machine is heavily contended (many ops want it), 
    # the effective completion time is longer, so we penalize using it.
    
    # Build a map: machine -> count of ready operations that can use it
    machine_demand = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_demand[m] = machine_demand.get(m, 0) + 1
    
    # Score each operation
    best_idx = 0
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        # For this op, find the best machine considering demand
        min_effective_time = float('inf')
        for m, dur in op["machine_options"].items():
            demand = machine_demand.get(m, 1)
            # Effective time: duration scaled by contention (higher demand = worse)
            # Use sqrt(demand) as a soft penalty to avoid over-penalizing
            effective_time = dur * (1.0 + 0.5 * (demand - 1))
            if effective_time < min_effective_time:
                min_effective_time = effective_time
        
        # Also consider remaining work as a tie-breaker: prefer ops with more remaining work
        # if effective times are close, because they're more critical
        rem_work = op.get("remaining_work", 0)
        
        # Composite score: minimize effective time, break ties by maximizing remaining work
        score = min_effective_time - 0.1 * rem_work
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)

# EVOLVE-BLOCK-END
