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
    machine_load_estimate = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_count[m] = machine_count.get(m, 0) + 1
            machine_load_estimate[m] = machine_load_estimate.get(m, 0.0) + float(d)
    
    # Precompute per-operation features
    urgency_scores = np.empty(n, dtype=float)
    min_durations = np.empty(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 0.0))
        opts = op["machine_options"]
        
        # Urgency: combine remaining work and operations
        urgency_scores[i] = rw + 0.1 * ro
        
        # Minimum duration for this operation
        if opts:
            min_durations[i] = min(opts.values())
        else:
            min_durations[i] = 0.0
    
    # Regret-based heuristic:
    # For each operation, compute the "regret" = difference between best and second-best
    # machine assignment costs. Operations with high regret have decisions that matter
    # most, so we prioritize them.
    
    # Cost for assigning op i to machine m:
    # cost(i, m) = duration(i, m) * (1 + CONTENT_WEIGHT * (contention(m) - 1))
    #            + FUTURE_LOAD_WEIGHT * estimated_future_load(m)
    #
    # Regret(i) = cost(i, second_best_m) - cost(i, best_m)
    #
    # We select the operation with the highest regret, breaking ties by urgency.
    
    CONTENT_WEIGHT = 0.3
    FUTURE_LOAD_WEIGHT = 0.2
    
    best_regret = -float('inf')
    best_index = 0
    best_machine = None
    best_urgency = -float('inf')
    
    for i, op in enumerate(ready_operations):
        urgency = urgency_scores[i]
        if urgency <= 0:
            urgency = 1e-6
        
        opts = op["machine_options"]
        if not opts:
            continue
        
        # Compute costs for all machines
        machine_costs = []
        for m, d in opts.items():
            d = float(d)
            if d <= 0:
                continue
            
            count = machine_count.get(m, 1)
            contention_factor = 1.0 + CONTENT_WEIGHT * (count - 1)
            est_completion = d * contention_factor
            
            # Future load: approximate by the total load on this machine
            # Normalize by number of machines to keep scale reasonable
            future_load = machine_load_estimate.get(m, 0.0) / max(1, len(machine_count))
            
            cost = est_completion + FUTURE_LOAD_WEIGHT * future_load
            machine_costs.append((cost, m))
        
        if len(machine_costs) == 0:
            continue
        
        # Sort by cost to find best and second-best
        machine_costs.sort(key=lambda x: x[0])
        best_cost, best_m = machine_costs[0]
        
        if len(machine_costs) >= 2:
            second_cost, _ = machine_costs[1]
            regret = second_cost - best_cost
        else:
            # Only one machine option: regret is low (no choice to regret)
            regret = 0.0
        
        # Select operation with highest regret; break ties by higher urgency
        if regret > best_regret + 1e-9:
            best_regret = regret
            best_index = i
            best_machine = best_m
            best_urgency = urgency
        elif abs(regret - best_regret) <= 1e-9:
            # Tie: pick higher urgency
            if urgency > best_urgency:
                best_regret = regret
                best_index = i
                best_machine = best_m
                best_urgency = urgency
    
    # Fallback if no valid score was found
    if best_machine is None:
        # Fall back to urgency-based selection
        best_index = 0
        best_urgency = -1
        for i, op in enumerate(ready_operations):
            if urgency_scores[i] > best_urgency:
                best_urgency = urgency_scores[i]
                best_index = i
        op = ready_operations[best_index]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: float(opts[k]))
        else:
            best_machine = None
    
    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
