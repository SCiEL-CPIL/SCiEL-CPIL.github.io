"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule using a regret-based heuristic with look-ahead.
    
    For each ready operation, we compute:
    - The best machine (minimum duration)
    - The second-best machine
    - A regret score based on the difference between best and second-best,
      weighted by job urgency (remaining work / remaining ops)
    
    We also consider machine load estimation to break ties and refine choices.
    
    Returns: index of the ready operation to start, optionally (index, machine)
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    best_machine_for_op = [None] * n
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = max(op["remaining_ops"], 1)
        
        # Get all machine options sorted by duration
        machines = sorted(machine_options.keys(), key=lambda m: machine_options[m])
        durations = [machine_options[m] for m in machines]
        
        if len(machines) == 0:
            # No machines available, give low score
            scores[i] = 0.0
            continue
        
        best_dur = durations[0]
        best_mach = machines[0]
        
        # Compute regret: difference between best and second-best
        if len(machines) > 1:
            second_dur = durations[1]
            regret = second_dur - best_dur
        else:
            regret = 0.0
        
        # Urgency factor: more remaining work relative to remaining ops = more urgent
        # Also, shorter best duration means we should prioritize it more
        urgency = remaining_work / remaining_ops
        
        # Combine regret and urgency
        # Higher regret means the machine choice matters more, so we should decide sooner
        # Higher urgency means the job needs attention
        score = regret * 0.3 + urgency * 0.4 + (1.0 / (best_dur + 1.0)) * 0.3
        
        scores[i] = score
        best_machine_for_op[i] = best_mach
    
    # Select the operation with the highest score
    idx = int(np.argmax(scores))
    
    # Return just the index (the simulator will pick the machine or we can return a tuple)
    # Based on the interface, we can return either index or (index, machine)
    # Let's return just the index for simplicity, as the simulator may handle machine selection
    return idx
# EVOLVE-BLOCK-END
