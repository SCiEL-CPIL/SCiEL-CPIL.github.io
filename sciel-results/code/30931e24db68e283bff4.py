# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using a Constraint-Based Bottleneck Heuristic.
    
    This heuristic aims to minimize makespan by:
    1. Prioritizing operations with fewer machine options (Least Eligible Machines).
       Operations with fewer options are more constrained and are more likely to
       become bottlenecks if delayed. This addresses the "machine-centric" aspect
       by focusing on the flexibility of the operation's assignment.
    2. Using Shortest Processing Time (SPT) as a tie-breaker. Among operations
       with the same level of constraint (same number of options), we pick the
       one with the shortest minimum processing time to reduce immediate machine
       occupation.
    
    Score for each ready operation:
        primary_score = len(machine_options)
        secondary_score = min(machine_options.values())
        
    We pick the operation with the lowest (primary_score, secondary_score) tuple.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Calculate scores
    # 1. Number of machine options (Constraint)
    # 2. Minimum duration among options (SPT)
    
    num_options = np.zeros(n, dtype=int)
    min_durations = np.zeros(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if opts:
            num_options[i] = len(opts)
            min_durations[i] = min(opts.values())
        else:
            # If no options, assign high priority (should not happen in valid input)
            num_options[i] = float('inf')
            min_durations[i] = float('inf')
            
    # We want to minimize num_options, then min_durations.
    # Using lexsort: last key is primary.
    # We want smallest num_options, then smallest min_durations.
    # So keys = (min_durations, num_options)
    
    indices = np.lexsort((min_durations, num_options))
    
    # The first index in the sorted array is the best one
    best_idx = int(indices[0])
    
    return best_idx
# EVOLVE-BLOCK-END
