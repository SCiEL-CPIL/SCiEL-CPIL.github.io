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
    
    Strategy: Model-and-optimize approach
    1. Calculate urgency based on remaining work and operations.
    2. Calculate machine load estimate based on duration and contention.
    3. Select operation-machine pair that minimizes a weighted combination of
       completion time estimate and job priority.
    
    The key insight is to prioritize jobs with more remaining work (to avoid
    starving long jobs) while selecting the machine that gives the earliest
    completion, accounting for machine contention.
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
    
    # Precompute machine contention: count how many ready ops can use each machine
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1
    
    # Precompute per-operation features
    urgency_scores = np.empty(n, dtype=float)
    min_durations = np.empty(n, dtype=float)
    regret_scores = np.empty(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 0.0))
        opts = op["machine_options"]
        
        # Urgency: combine remaining work and operations
        urgency_scores[i] = rw + 0.1 * ro
        
        # Compute regret: difference between best and second-best machine assignment
        # Higher regret means the machine choice matters more
        if opts and len(opts) >= 2:
            sorted_opts = sorted(opts.items(), key=lambda x: float(x[1]))
            best_cost = float(sorted_opts[0][1])
            second_cost = float(sorted_opts[1][1])
            # Regret is the cost of delaying this decision (difference in completion)
            # Scale by contention to account for future load
            best_m = sorted_opts[0][0]
            best_count = machine_count.get(best_m, 1)
            regret_scores[i] = (second_cost - best_cost) * (1.0 + 0.3 * (best_count - 1))
        elif opts:
            regret_scores[i] = 0.0
        else:
            regret_scores[i] = 0.0
            min_durations[i] = 0.0
            continue
        
        # Minimum duration for this operation
        min_durations[i] = min(opts.values())
    
    # Now evaluate each operation-machine pair
    # Score combines urgency, regret, and completion time estimate
    # Regret prioritizes decisions where machine choice has high impact
    
    best_score = -float('inf')
    best_index = 0
    best_machine = None
    
    # Weight parameters
    URG_WEIGHT = 1.0
    REGRET_WEIGHT = 0.5
    CONTENT_WEIGHT = 0.3
    
    for i, op in enumerate(ready_operations):
        urgency = urgency_scores[i]
        if urgency <= 0:
            urgency = 1e-6
        
        regret = regret_scores[i]
        
        for m, d in op["machine_options"].items():
            d = float(d)
            if d <= 0:
                continue
            
            # Machine contention penalty
            count = machine_count.get(m, 1)
            contention_factor = 1.0 + CONTENT_WEIGHT * (count - 1)
            
            # Estimated completion time
            est_completion = d * contention_factor
            
            # Score: higher is better
            # Combine urgency, regret (prioritize high-regret decisions), and low completion
            score = URG_WEIGHT * urgency / (est_completion + 1e-9)
            score += REGRET_WEIGHT * regret / (urgency + 1e-9)
            
            if score > best_score:
                best_score = score
                best_index = i
                best_machine = m
    
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
