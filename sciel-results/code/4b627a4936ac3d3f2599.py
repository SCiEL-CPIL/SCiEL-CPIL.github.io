"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 115.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Strategy: Critical Path / WSJF Hybrid
    1. Calculate job urgency as the sum of minimum durations of all remaining operations.
       This approximates the critical path length of the remaining job.
    2. Estimate machine cost as duration adjusted for contention (number of other ready ops 
       that can use that machine).
    3. Select the operation-machine pair that maximizes Urgency / Cost.
       This prioritizes jobs that are "long" (critical) and can be finished quickly on a 
       less contended machine.
    """
    if _elapsed() > _DEADLINE:
        # Hard safety fallback: Shortest Processing Time (SPT)
        if not ready_operations:
            return 0
        best_i = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
        return best_i

    if not ready_operations:
        return 0
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    n = len(ready_operations)
    
    # Precompute machine load estimates: sum of min durations of all ready ops that can use each machine
    # This approximates future utilization of each machine
    machine_load = {}
    for op in ready_operations:
        opts = op["machine_options"]
        if not opts:
            continue
        min_d = min(opts.values())
        for m in opts:
            machine_load[m] = machine_load.get(m, 0.0) + min_d
    
    # Precompute per-operation urgency
    urgency_scores = np.empty(n, dtype=float)
    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        urgency_scores[i] = rw if rw > 0 else 1.0

    best_score = -float('inf')
    best_index = 0
    best_machine = None
    
    # Strategy: Load-balancing with critical path awareness
    # For each operation-machine pair, estimate the resulting maximum machine load
    # (bottleneck) after assigning this operation to that machine.
    # Select the pair that minimizes the bottleneck, weighted by job urgency.
    
    for i, op in enumerate(ready_operations):
        urgency = urgency_scores[i]
        if urgency <= 0:
            urgency = 1e-6
            
        opts = op["machine_options"]
        if not opts:
            continue

        # Find the machine that minimizes the resulting bottleneck load
        # when this operation is assigned to it
        best_m_for_op = None
        best_bottleneck = float('inf')
        
        for m, d in opts.items():
            d = float(d)
            if d <= 0:
                continue
            
            # Current estimated load on machine m (excluding this op)
            current_load = machine_load.get(m, 0.0)
            
            # New load on machine m if we assign this op to it
            new_load = current_load + d
            
            # The bottleneck is the max load across all machines after this assignment
            # Approximate by considering only the load on m and the max of other machines
            # To be more accurate, we'd need to track all machine loads, but for efficiency:
            # The key insight is to avoid making any machine the new maximum if possible
            
            # Estimate the resulting max load: max(new_load, max_load_of_other_machines)
            # We approximate max_load_of_other_machines by the current max load
            # excluding this machine's contribution
            
            # For simplicity, use a soft-max approximation:
            # The bottleneck is dominated by the largest load
            # We want to minimize the increase in the global maximum
            
            # Alternative: minimize new_load directly, with a penalty if it exceeds
            # the current average load significantly
            
            # Let's use a simpler approach: minimize the ratio of new_load to current_load
            # This favors machines that are less loaded relative to their current usage
            # But if current_load is 0, we need to handle that
            
            if current_load > 1e-9:
                # Relative load increase
                rel_increase = new_load / current_load
            else:
                # Machine is currently idle; prefer it if d is small
                rel_increase = d + 1e-9  # Small penalty for idle machines to avoid over-penalizing
            
            # Combine with absolute load to prevent extreme imbalances
            # Use a weighted combination
            score_component = 0.7 * rel_increase + 0.3 * (new_load + 1e-9)
            
            if score_component < best_bottleneck:
                best_bottleneck = score_component
                best_m_for_op = m
        
        if best_m_for_op is None:
            continue
            
        # Final score: prioritize high urgency jobs, penalize high bottleneck
        # Use WSJF-like: Urgency / Bottleneck
        score = urgency / (best_bottleneck + 1e-9)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Fallback if no valid score was found
    if best_machine is None:
        best_index = 0
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: opts[k])
        else:
            best_machine = None
    
    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
