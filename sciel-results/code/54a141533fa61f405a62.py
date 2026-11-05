"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s wall-clock limit


def _elapsed():
    return time.time() - _start_time


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling using bipartite matching-inspired greedy heuristic.
    
    Strategy:
    Model ready operations as edges in a bipartite graph (jobs-machines).
    Select the operation with highest processing time on the machine with fewest total remaining workload,
    balancing load while prioritizing critical long operations.
    
    Score for each operation:
    - Primary: processing time on the least-loaded machine (max weight matching edge)
    - Secondary: job's remaining work (prioritize jobs with more work)
    - Tertiary: fewer remaining ops (finish jobs sooner)
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return 0
    
    # Compute total remaining workload per machine across all ready operations
    machine_load = {}
    for op in ready_operations:
        for machine, dur in op["machine_options"].items():
            if machine not in machine_load:
                machine_load[machine] = 0.0
            machine_load[machine] += dur
    
    # Compute scores for each ready operation
    scores = []
    for op in ready_operations:
        work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Find the best machine for this operation: highest duration / lowest machine load
        best_ratio = -1.0
        best_dur = 0.0
        for machine, dur in op["machine_options"].items():
            load = machine_load.get(machine, 0.0)
            # Ratio: prefer longer durations on less-loaded machines
            # Add small epsilon to avoid division by zero
            ratio = dur / (load + 1.0)
            if ratio > best_ratio:
                best_ratio = ratio
                best_dur = dur
        
        # Composite score:
        # - Primary: best_ratio (load-balanced processing time)
        # - Secondary: remaining_work (prioritize jobs with more work)
        # - Tertiary: -remaining_ops (finish jobs with fewer ops left)
        
        score = best_ratio * 1000.0 + work * 10.0 - remaining_ops * 1.0
        
        scores.append(score)
    
    # Return index of operation with highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
