"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Global deadline guard
_start_time = time_module.time()
_DEADLINE = 145.0  # 5 seconds margin from 150s limit

def dispatch(ready_operations, time):
    # Check deadline
    if time_module.time() - _start_time > _DEADLINE:
        # Fallback to simple index 0 or best effort
        return 0

    if not ready_operations:
        return 0

    # Heuristic: Shortest Processing Time (SPT) on the best machine option,
    # tie-broken by Most Work Remaining (MWR).
    # This targets machine-centric load balancing by prioritizing operations
    # that can be completed quickly, reducing queue lengths and contention.
    
    min_durations = []
    remaining_works = []
    
    for op in ready_operations:
        # Find the minimum duration across eligible machines
        # machine_options is {machine: duration}
        min_dur = min(op["machine_options"].values())
        min_durations.append(min_dur)
        remaining_works.append(op["remaining_work"])
        
    min_durations = np.array(min_durations)
    remaining_works = np.array(remaining_works)
    
    # We want to minimize min_duration.
    # If there are ties, we want to maximize remaining_work.
    # To do this with argmin/argmax, we can create a composite score.
    # score = min_duration - epsilon * remaining_work
    # Since min_duration is usually larger than 1, and remaining_work is an integer,
    # we need to scale them appropriately or use lexicographic sorting.
    
    # Lexicographic sort is safer.
    # Sort indices by (min_duration, -remaining_work)
    indices = np.arange(len(ready_operations))
    
    # Create a structured array or use np.lexsort
    # np.lexsort sorts by the last key first.
    # We want primary key: min_durations (ascending)
    # Secondary key: -remaining_works (ascending, which means max remaining_work)
    sorted_indices = np.lexsort((-remaining_works, min_durations))
    
    return int(sorted_indices[0])
# EVOLVE-BLOCK-END
