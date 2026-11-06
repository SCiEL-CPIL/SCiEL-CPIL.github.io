"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule based on machine-contention avoidance.
    
    For each ready operation, we compute a score that favors operations
    that can be executed on less loaded machines. The idea is to use the
    ratio of the minimum required duration to the sum of durations across
    all available machines for that operation. This prioritizes operations
    that have more flexibility in machine assignment (i.e., can be run on
    machines with lower contention).
    
    We also incorporate remaining work to break ties and add some priority
    to jobs with more remaining work.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        
        if not machine_options:
            scores[i] = 0.0
            continue
        
        # Get all durations for this operation across available machines
        durations = list(machine_options.values())
        
        if not durations:
            scores[i] = 0.0
            continue
        
        min_duration = min(durations)
        sum_durations = sum(durations)
        
        # Contention avoidance score: ratio of min to sum duration
        # Higher ratio means the operation is less sensitive to which machine it uses
        # (more balanced durations), so it's safer to schedule
        contention_ratio = min_duration / sum_durations if sum_durations > 0 else 0.0
        
        # Also consider the number of machine options - more options means more flexibility
        n_machines = len(machine_options)
        
        # Combine contention ratio with remaining work
        # We want to prioritize operations that:
        # 1. Have less machine contention (higher contention_ratio)
        # 2. Have more remaining work (to reduce makespan)
        
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Normalize scores by combining factors
        # Use a weighted combination:
        # - contention_ratio favors operations with flexible machine choices
        # - remaining_work favors operations with more work remaining
        # - n_machines gives a small bonus for more options
        
        # Scale remaining work to be comparable to contention_ratio
        # Use a simple heuristic: weight contention_ratio heavily, with remaining work as tiebreaker
        scores[i] = contention_ratio + 0.1 * (remaining_work / 100.0) + 0.05 * np.log1p(n_machines)
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
