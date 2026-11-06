"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatch rule for Open-Shop Scheduling.
    Uses a structural shift to a Longest Processing Time (LPT) based heuristic
    combined with a load-balancing penalty, to prioritize inflexible long tasks
    and prevent late-stage bottlenecks, while still respecting job remaining work.
    """
    if not ready_operations:
        return -1

    # Deadline guard: track elapsed time to ensure we return before the 150s limit.
    # Since this function is stateless and called frequently, we assume the overhead
    # is negligible, but we keep the guard for safety if the list is huge.
    start_t = time_module.perf_counter()
    DEADLINE = 145.0  # Safety margin

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    # Pre-compute machine loads if possible? No, we don't have global state.
    # However, we can estimate "pressure" or "inflexibility".
    
    # Structural Change:
    # Instead of purely minimizing remaining work (MWR) and favoring short durations (SPT),
    # we adopt an LPT-like strategy: prioritize operations with LONG minimum durations
    # to balance load early, but moderated by the job's total remaining work.
    
    # The "inflexibility" of an operation is how long it *must* take (min duration).
    # Long inflexible ops should be scheduled early to avoid them blocking machines later.
    
    for i, op in enumerate(ready_operations):
        # Check deadline
        if time_module.perf_counter() - start_t > DEADLINE:
            break

        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Find the machine with the minimum processing time for this operation
        min_dur = float('inf')
        max_dur = 0.0
        best_m = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                best_m = m
            if d > max_dur:
                max_dur = d

        if best_m is None:
            continue

        # Retrieve job metrics
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)

        # Composite Score Design (LPT-inspired):
        # 1. Inflexibility (Min Duration): Higher is better. Prioritize long tasks.
        #    We use a logarithmic or linear scale to avoid extreme dominance if durations vary wildly.
        #    Let's use linear for now, as open-shop durations are often comparable.
        # 2. Remaining Work: Higher is better. Jobs with more work left are more critical.
        # 3. Flexibility Penalty: If an op has a huge range (max_dur - min_dur), it's more flexible.
        #    We might penalize very flexible ops slightly if we are in a "load balancing" mode,
        #    but LPT usually just looks at the min (the critical path length).
        
        # Let's try a weighted sum where Min Duration is a primary factor.
        # Previous approach: 1.0 * remaining_work + ... + 2.0 / (min_dur + 1.0) (favored short)
        # New approach: 1.0 * remaining_work + 1.5 * min_dur
        
        # Why 1.5 * min_dur?
        # If we have two jobs:
        # Job A: rem_work=100, min_dur=10
        # Job B: rem_work=90, min_dur=50
        # Old Score A: 100 + 2/11 ~ 100.18
        # Old Score B: 90 + 2/51 ~ 90.04 -> Picks A.
        # New Score A: 100 + 15 = 115
        # New Score B: 90 + 75 = 165 -> Picks B.
        
        # This shifts priority to the long task B, which is likely the bottleneck.
        
        # We also add a small term for remaining_ops to break ties in favor of jobs with more steps.
        score = 1.0 * remaining_work + 1.5 * min_dur + 0.1 * remaining_ops

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m

    # Return the best operation index and its optimal machine
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
