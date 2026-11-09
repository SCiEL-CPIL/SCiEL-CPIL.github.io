"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatches the next operation to start.
    
    Args:
        ready_operations: List of dicts, each with 'job', 'op', 'machine_options' {machine: duration}, 
                          'remaining_ops', 'remaining_work'.
        time: Current simulation time.
    
    Returns:
        index: Index of the ready operation to start.
        machine: (Optional) The machine to use.
    """
    if not ready_operations:
        return 0

    best_score = -1
    best_index = 0
    best_machine = None

    # Deadline guard (though computation is trivial, good practice per directive)
    start_time = time_module.time()
    deadline = start_time + 0.1  # 100ms is plenty for this small list

    for i, op in enumerate(ready_operations):
        if time_module.time() > deadline:
            # Fallback to a simple heuristic if we run out of time (unlikely)
            break
            
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Find the machine with the shortest duration for this operation
        # This minimizes the time the operation occupies a machine
        min_duration = float('inf')
        chosen_machine = None
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                chosen_machine = m
        
        if chosen_machine is None:
            continue

        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        
        # Heuristic: Prioritize operations with high remaining work and ops,
        # but penalize those with long processing times on the best machine.
        # Score = (Remaining Work + 0.5 * Remaining Ops) / (Processing Time + 1)
        # This ratio favors "dense" work that can be cleared quickly.
        denominator = min_duration + 1.0
        score = (rw + 0.5 * ro) / denominator
        
        # Add a small bonus for jobs with more remaining ops to prevent starvation
        # of long multi-step jobs
        score += 0.1 * ro

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = chosen_machine

    # If no valid machine was found or loop broke early, return best found or 0
    if best_machine is not None:
        return (best_index, best_machine)
    else:
        return best_index
# EVOLVE-BLOCK-END
