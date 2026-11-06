"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Greedy LPT approximation for makespan minimization.
    
    Strategy:
    1. Prioritize operations with the longest minimum processing time (max min_duration).
       This ensures that long, potentially bottlenecking tasks are started early.
    2. Break ties by favoring jobs with more total remaining work.
    3. Break further ties by favoring jobs with more remaining operations.
    """
    if not ready_operations:
        return 0
    
    # Extract key features for each ready operation
    min_durations = []
    remaining_works = []
    remaining_ops_counts = []
    
    for op in ready_operations:
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        
        # If no machine options, treat as 0 duration (should not happen in valid instances)
        min_dur = min(durations) if durations else 0
        min_durations.append(min_dur)
        
        # Use remaining work as secondary priority
        rem_work = op["remaining_work"]
        remaining_works.append(rem_work)
        
        # Use remaining ops as tertiary priority
        rem_ops = op["remaining_ops"]
        remaining_ops_counts.append(rem_ops)
    
    # Convert to numpy arrays for vectorized comparison
    min_durations = np.array(min_durations)
    remaining_works = np.array(remaining_works)
    remaining_ops_counts = np.array(remaining_ops_counts)
    
    # To implement lexicographic ordering (max min_dur, then max rem_work, then max rem_ops):
    # We can create a composite score where the primary key dominates.
    # However, scales of min_duration and remaining_work might differ.
    # A safer approach for "max min_duration" primarily is to find the max min_duration,
    # then filter candidates, then apply next criteria.
    
    # Step 1: Find the maximum minimum duration
    max_min_dur = np.max(min_durations)
    
    # Step 2: Filter operations that have this maximum minimum duration
    # We want indices where min_duration == max_min_dur
    # Due to floating point, we might use tolerance, but integers are likely.
    candidates = np.where(np.isclose(min_durations, max_min_dur))[0]
    
    if len(candidates) == 1:
        return int(candidates[0])
    
    # Step 3: Among candidates, pick the one with max remaining_work
    candidate_works = remaining_works[candidates]
    max_work = np.max(candidate_works)
    
    # Filter again for max work
    work_candidates = candidates[np.where(np.isclose(candidate_works, max_work))[0]]
    
    if len(work_candidates) == 1:
        return int(work_candidates[0])
    
    # Step 4: Among those, pick the one with max remaining_ops
    candidate_ops = remaining_ops_counts[work_candidates]
    max_ops = np.max(candidate_ops)
    
    # Filter again for max ops
    op_candidates = work_candidates[np.where(np.isclose(candidate_ops, max_ops))[0]]
    
    if len(op_candidates) == 1:
        return int(op_candidates[0])
    
    # Step 5: If still tied, pick the first one (or arbitrary)
    return int(op_candidates[0])
# EVOLVE-BLOCK-END
