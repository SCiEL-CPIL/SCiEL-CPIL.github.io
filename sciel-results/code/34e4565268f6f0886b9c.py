"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Pick the next ready operation to start in an open-shop simulation.
    
    Strategy: Balance machine loads by preferring operations on the most
    congested machines, with a tie-breaker favoring shorter durations on
    those machines to reduce contention. Also considers remaining work
    per job to avoid starving jobs.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're taking too long, fall back to simple heuristic
    start_time = _time.time()
    deadline = start_time + 0.01  # 10ms budget per dispatch call

    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0

    # Quick heuristic: if only a few ops, use combined score
    # Compute for each op:
    #   - machine load: sum of remaining_work across all ops targeting same machine
    #   - job remaining: op["remaining_work"]
    #   - duration on chosen machine
    
    # For each ready operation, we need to decide which machine to assign it to.
    # The machine_options dict maps machine -> duration.
    
    # First, compute total remaining workload per machine from ready_operations
    # Note: we only have ready operations, not all pending operations.
    # So we approximate machine congestion from ready ops only.
    
    machine_load = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            if m not in machine_load:
                machine_load[m] = 0.0
            # Each ready op contributes its duration on that machine to load
            machine_load[m] += op["machine_options"][m]
    
    # For each operation, find the best machine to assign it to
    # Score for an op on machine m:
    #   Primary: higher machine_load[m] is better (balance load)
    #   Secondary: shorter duration is better (less blocking)
    #   Tertiary: more remaining_work for the job (don't starve)
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        job_rem = op["remaining_work"]
        machine_opts = op["machine_options"]
        
        # Check deadline
        if _time.time() - start_time > deadline:
            # Fallback: simple remaining work heuristic
            break
        
        for m, dur in machine_opts.items():
            m_load = machine_load.get(m, 0.0)
            
            # Combined score:
            # - Weight machine load (congestion) heavily
            # - Subtract duration (shorter is better on busy machines)
            # - Add small bonus for job remaining work
            score = 2.0 * m_load - 0.5 * dur + 0.1 * job_rem
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = m
    
    # If deadline was hit, use simple fallback
    if _time.time() - start_time > deadline:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    return int(best_idx)
# EVOLVE-BLOCK-END
