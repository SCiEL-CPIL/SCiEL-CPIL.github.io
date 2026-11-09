"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using Normalized Rank Combination.
    
    Instead of using raw ratios which can be skewed by outliers or different scales,
    we rank the operations by two key metrics:
    1. Min Processing Time (Shorter is better -> Lower rank)
    2. Remaining Work (More is better -> Lower rank)
    
    We combine these ranks to select the operation. This approach is robust to
    scale differences between duration and work values.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0

    min_durations = []
    remaining_works = []
    
    for op in ready_operations:
        machine_options = op["machine_options"]
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 0.0
            
        remaining_work = op["remaining_work"]
        
        min_durations.append(min_duration)
        remaining_works.append(remaining_work)

    min_durations = np.array(min_durations)
    remaining_works = np.array(remaining_works)
    
    # Rank: 1 is best.
    # For min_duration: lower value is better. 
    # argsort gives indices that would sort the array. 
    # We want the rank of each element.
    # np.argsort(np.argsort(x)) gives ranks (0-based).
    
    # Rank min_duration: ascending order is better.
    rank_duration = np.argsort(np.argsort(min_durations))
    
    # Rank remaining_work: descending order is better (more work = more urgent/critical).
    # To get descending rank, we can negate the array before ranking, or rank ascending and flip.
    rank_work = np.argsort(np.argsort(-remaining_works))
    
    # Combine ranks.
    # Weighting: Usually, critical path (work) is slightly more important than immediate step duration.
    # But in JSS, blocking is a big issue. Let's try equal weight or slight bias to duration.
    # Let's try 0.5 * rank_duration + 0.5 * rank_work
    
    scores = 0.5 * rank_duration + 0.5 * rank_work
    
    # Pick the operation with the lowest combined score (best ranks)
    best_index = int(np.argmin(scores))
    
    return best_index
# EVOLVE-BLOCK-END
