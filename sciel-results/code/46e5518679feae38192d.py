"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    """
    Machine-load balancing strategy:
    Select the ready operation with minimum remaining workload on its most
    constrained machine (highest contention / fewest options) to prevent
    any single machine from becoming the dominant bottleneck.
    
    Heuristic:
    1. For each machine, compute a contention score based on how many ready
       operations can run on it (higher = more constrained).
    2. For each ready operation, identify its most constrained machine among
       its machine_options.
    3. Score each operation: prefer operations that have the MINIMUM duration
       on their most constrained machine (to clear the bottleneck fastest),
       then tie-break by least total remaining work.
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Count how many ready operations can run on each machine (contention)
    machine_contention = defaultdict(int)
    for op in ready_operations:
        for machine in op["machine_options"]:
            machine_contention[machine] += 1
    
    best_score = float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        
        if not machine_options:
            continue
        
        # Find the most constrained machine for this operation
        # (highest contention among its options)
        max_contention = max(machine_contention[m] for m in machine_options)
        
        # Among machines with max contention, pick the one with shortest duration
        # This is the key: we want to clear the most contested machine fastest
        min_duration_on_constrained = min(
            machine_options[m] for m in machine_options 
            if machine_contention[m] == max_contention
        )
        
        # Score: lower is better
        # Primary: shortest duration on the most contested machine
        # Secondary: least total remaining work
        # Tertiary: operations with more remaining operations get slight priority
        # (to avoid starvation)
        remaining_ops = op["remaining_ops"]
        
        score = min_duration_on_constrained + remaining_work * 0.01 + remaining_ops * 0.001
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
