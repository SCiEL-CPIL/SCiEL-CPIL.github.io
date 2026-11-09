"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Greedy Bottleneck Elimination (LPT-style) dispatching rule.
    
    For each ready operation, compute the minimum processing time across
    all available machine options. Prioritize the operation with the
    MAXIMUM such minimum time (i.e., the most inflexible/longest bottleneck
    operation). This aggressively schedules long, inflexible operations
    early to prevent them from becoming critical path bottlenecks later.
    
    Ties are broken by:
    1. Fewer machine_options (higher constraint/inflexibility)
    2. Higher remaining_work (more total work left in the job)
    3. Fewer remaining_ops (closer to completion, to avoid idle gaps)
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute metrics for each ready operation
    # We use a lexicographic sort key:
    # 1. -min_time (Maximize min_time)
    # 2. num_options (Minimize num_options -> Prefer constrained)
    # 3. -remaining_work (Maximize remaining_work)
    # 4. remaining_ops (Minimize remaining_ops)
    # 5. index (Stable tie-breaker)
    
    best_score = None
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        min_time = min(durations) if durations else 0
        num_options = len(durations)
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        score = (
            -min_time,
            num_options,
            -remaining_work,
            remaining_ops,
            i
        )
        
        if best_score is None or score < best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
