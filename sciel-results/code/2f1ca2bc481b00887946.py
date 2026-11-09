"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using a Constraint-Based Heuristic.
    
    This approach prioritizes operations that are:
    1. Urgent (high remaining work)
    2. Constrained (few machine options, i.e., low flexibility)
    3. Short (low processing time on the best machine)
    
    The score is a multiplicative combination of Urgency and Constraint,
    with a penalty for Processing Time. This differs from additive static
    weights by emphasizing the interaction between job urgency and machine
    flexibility.
    """
    if not ready_operations:
        return 0
    
    best_score = -float('inf')
    best_index = 0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        num_machines = len(machine_options)
        durations = list(machine_options.values())
        min_duration = min(durations)
        
        # Constraint Factor: Higher for fewer machine options (more constrained)
        # Using 1.0 / num_machines ensures that single-machine options get max weight.
        constraint_factor = 1.0 / num_machines
        
        # Score: Urgency * Constraint - Penalty for Duration
        # We multiply remaining work by constraint factor to prioritize
        # critical jobs that have limited routing options.
        # We subtract a small fraction of the minimum duration to break ties
        # in favor of shorter operations (SPT) to reduce machine blocking.
        
        score = remaining_work * constraint_factor - 0.1 * min_duration
        
        if score > best_score:
            best_score = score
            best_index = i
            
    return best_index
# EVOLVE-BLOCK-END
