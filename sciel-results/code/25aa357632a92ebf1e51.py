"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Uses a composite heuristic that balances:
    1. Most work remaining (MW) - to ensure large jobs get priority
    2. Earliest completion time (ECT) - to minimize makespan by finishing operations quickly
    3. Machine load balancing - to avoid bottlenecks
    
    The score for each ready operation is a weighted combination of:
    - Normalized remaining work (higher is better)
    - Negative minimum duration across machines (lower duration is better)
    - A penalty if the machine with the shortest duration is heavily loaded (approximated by time)
    
    Returns the index of the best operation to start next.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Calculate scores for each ready operation
    scores = np.zeros(n)
    
    # Pre-compute max remaining work for normalization
    max_remaining_work = max(op["remaining_work"] for op in ready_operations)
    if max_remaining_work == 0:
        max_remaining_work = 1
    
    # Pre-compute min durations for normalization
    min_durations = []
    for op in ready_operations:
        if op["machine_options"]:
            min_durations.append(min(op["machine_options"].values()))
        else:
            min_durations.append(0)
    max_min_duration = max(min_durations) if min_durations else 1
    if max_min_duration == 0:
        max_min_duration = 1
    
    for i, op in enumerate(ready_operations):
        # Component 1: Remaining work (higher is better)
        work_score = op["remaining_work"] / max_remaining_work
        
        # Component 2: Shortest possible duration (lower is better, so negate)
        if op["machine_options"]:
            min_dur = min(op["machine_options"].values())
        else:
            min_dur = 0
        duration_score = 1.0 - (min_dur / max_min_duration)
        
        # Component 3: Number of remaining ops (more remaining ops means more flexibility)
        # Actually, fewer remaining ops might be better to complete jobs sooner
        # Let's use: prefer ops that are last in their job (to complete jobs)
        # But we don't have direct info on position, so use remaining_ops count
        # Fewer remaining ops = closer to completion, which is good
        remaining_ops = op["remaining_ops"]
        # Normalize: assume max remaining ops is small (10 jobs, 5 machines, so max ~5-10)
        # Use a simple scaling
        completion_score = 1.0 / (1.0 + remaining_ops)
        
        # Weighted combination
        # Weights: emphasize work remaining and duration
        weights = np.array([0.4, 0.4, 0.2])
        scores[i] = (weights[0] * work_score + 
                     weights[1] * duration_score + 
                     weights[2] * completion_score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
