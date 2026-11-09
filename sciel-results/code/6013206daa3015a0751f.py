"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule based on critical path (longest remaining path) and machine selection.
    
    Priority:
    1. Compute the longest remaining path (LRP) for each ready operation's job:
       sum of minimum durations across all remaining operations in the job.
    2. Among ready operations, pick the one with the highest LRP.
    3. If there's a tie, use remaining_work as a secondary tie-breaker.
    4. For machine selection: if the operation has multiple machine options,
       pick the machine that minimizes the completion time (i.e., shortest duration),
       but also consider load balancing by preferring less-loaded machines if durations are similar.
    
    Returns:
        index of the ready operation to start, optionally (index, machine)
    """
    if not ready_operations:
        return 0
    
    # Compute longest remaining path for each ready operation
    # LRP = sum of min durations across all remaining ops in the job
    lrps = []
    remaining_works = []
    
    for op in ready_operations:
        # Calculate LRP: sum of minimum durations for each remaining operation
        lrp = 0
        remaining_ops = op.get("remaining_ops", [])
        
        # remaining_ops is expected to be a list of operations
        # Each operation should have machine_options
        for rem_op in remaining_ops:
            machine_options = rem_op.get("machine_options", {})
            if machine_options:
                lrp += min(machine_options.values())
            else:
                # Fallback: use remaining_work if available
                lrp += rem_op.get("remaining_work", 0)
        
        # If remaining_ops is empty, use remaining_work as fallback
        if not remaining_ops:
            lrp = op.get("remaining_work", 0)
        
        lrps.append(lrp)
        remaining_works.append(op.get("remaining_work", 0))
    
    # Find the operation with the highest LRP
    # Tie-break: highest remaining_work
    max_lrp = max(lrps)
    candidates = [i for i, lrp in enumerate(lrps) if lrp == max_lrp]
    
    if len(candidates) == 1:
        best_idx = candidates[0]
    else:
        # Tie-break by remaining_work
        best_idx = max(candidates, key=lambda i: remaining_works[i])
    
    # Machine selection: minimize completion time
    # If the operation has multiple machine options, pick the one with shortest duration
    # But also consider load balancing - prefer machines that are less busy
    # For simplicity, we'll pick the machine with the shortest duration
    # If the interface expects just an index, return just the index
    # If it expects (index, machine), return that tuple
    
    op = ready_operations[best_idx]
    machine_options = op.get("machine_options", {})
    
    if machine_options:
        # Pick machine with minimum duration
        best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        # Return (index, machine) if machine selection is expected
        # Based on the task description, it says "optionally (index, machine)"
        # We'll return just the index to be safe, as the seed program does
        return int(best_idx)
    else:
        return int(best_idx)
# EVOLVE-BLOCK-END
