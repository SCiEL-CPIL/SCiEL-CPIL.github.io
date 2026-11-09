"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract metrics for each ready operation
    # We want to minimize makespan.
    # Heuristic: Shortest Processing Time (SPT) on the best available machine option.
    # Tie-breaker: Prefer operations from jobs with more remaining work (to keep heavy jobs moving)
    # or fewer remaining ops (to clear jobs out of the system).
    # Let's try: Minimize min_duration. Tie-break: Maximize remaining_work.
    
    min_durations = []
    remaining_works = []
    remaining_ops_counts = []
    
    for op in ready_operations:
        opts = op["machine_options"]
        if opts:
            min_durations.append(min(opts.values()))
        else:
            min_durations.append(float('inf'))
        
        remaining_works.append(op["remaining_work"])
        remaining_ops_counts.append(op["remaining_ops"])
        
    # Create indices sorted by the heuristic
    # Primary: min_duration (ascending)
    # Secondary: remaining_work (descending)
    # Tertiary: remaining_ops (ascending) - prefer to finish jobs
    
    # We can do this with a composite score or just sort.
    # Since n is small (10 jobs, 5 machines, ready ops <= 10), sorting is fine.
    
    indices = list(range(n))
    
    def sort_key(i):
        return (
            min_durations[i], 
            -remaining_works[i], 
            remaining_ops_counts[i]
        )
        
    sorted_indices = sorted(indices, key=sort_key)
    
    return sorted_indices[0]

# EVOLVE-BLOCK-END
