"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Deadline guard
START_TIME = time_module.time()
TIME_LIMIT = 140  # 10s margin from 150s limit

def dispatch(ready_operations, time):
    """
    Dispatch rule for Open-Shop Scheduling.
    
    Strategy:
    1. Prioritize jobs with the most remaining work (Critical Path / MWR).
    2. Tie-breaker: Prefer operations with shorter processing time (SPT) to keep machines free and flexible.
    3. Machine Selection: If multiple machines are available for an operation, choose the one with the shortest duration to minimize machine occupation time.
    
    This is a structural change from pure MWR to a composite heuristic that accounts for machine flexibility (via SPT tie-breaking and machine selection).
    """
    if not ready_operations:
        return 0

    # Check deadline
    if time_module.time() - START_TIME > TIME_LIMIT:
        return 0

    best_idx = 0
    best_score = -float('inf')
    best_machine = None

    # Pre-calculate durations for quick access
    # We need to pick the best machine for each op first
    op_details = []
    for i, op in enumerate(ready_operations):
        job = op['job']
        remaining_work = op['remaining_work']
        remaining_ops = op['remaining_ops']
        
        # Select best machine: Shortest duration
        min_duration = float('inf')
        best_m = None
        for m, d in op['machine_options'].items():
            if d < min_duration:
                min_duration = d
                best_m = m
        
        # Heuristic Score:
        # Primary: Remaining Work (Higher is better)
        # Secondary: Shorter Duration (Lower is better) -> Invert or subtract
        # Tertiary: More Remaining Ops (Higher is better) -> Complexity
        
        # We want to maximize this score
        # score = remaining_work - 0.1 * min_duration + 0.01 * remaining_ops
        score = remaining_work - 0.1 * min_duration + 0.01 * remaining_ops
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m

    return best_idx
# EVOLVE-BLOCK-END
