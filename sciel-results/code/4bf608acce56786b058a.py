"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Shortest Processing Time (SPT) on best machine,
    with Most Work Remaining (MWR) as tie-breaker.
    Returns (index, machine) to explicitly select the fastest machine.
    """
    if not ready_operations:
        return 0 # Fallback, though shouldn't happen

    best_score = float('inf')
    best_idx = 0
    best_machine = None
    best_remaining_work = -1

    for i, op in enumerate(ready_operations):
        # Find the minimum duration and corresponding machine
        min_dur = float('inf')
        min_machine = None
        for machine, dur in op["machine_options"].items():
            if dur < min_dur:
                min_dur = dur
                min_machine = machine
        
        # Heuristic: Minimize processing time (SPT)
        # Tie-breaker: Maximize remaining work (MWR)
        # We can combine these into a single score or use lexicographic ordering.
        
        # Lexicographic: Primary key = min_dur (ascending), Secondary key = -remaining_work (ascending)
        # Since we are iterating, we can just compare.
        
        current_remaining_work = op["remaining_work"]
        
        if min_dur < best_score:
            best_score = min_dur
            best_idx = i
            best_machine = min_machine
            best_remaining_work = current_remaining_work
        elif min_dur == best_score:
            # Tie-breaker: Prefer job with more remaining work
            if current_remaining_work > best_remaining_work:
                best_idx = i
                best_machine = min_machine
                best_remaining_work = current_remaining_work
                # best_score remains same, but we updated the chosen op

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
