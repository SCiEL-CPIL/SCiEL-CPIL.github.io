"""Regret-based look-ahead dispatch for flexible job-shop scheduling."""
import time as _time
import numpy as np

_DEADLINE = 145.0
_T0 = None

def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0

def _estimate_machine_load(ready_ops, machine, weights):
    """
    Estimate future load on a machine based on other ready operations
    that could also use this machine.
    """
    total_load = 0.0
    for op in ready_ops:
        opts = op["machine_options"]
        if machine in opts:
            # Use a fraction of the duration to represent potential load
            # The factor depends on how many machines this op has
            n_opts = len(opts)
            if n_opts > 0:
                total_load += float(opts[machine]) / n_opts
    return total_load

def _compute_regret(op, ready_ops, weights):
    """
    Compute the regret for an operation: the difference between its
    best and second-best machine assignment costs, where cost includes
    duration plus estimated future machine load.
    """
    opts = op["machine_options"]
    if not opts:
        return 0.0, None, None
    
    # Compute cost for each machine: duration + estimated future load
    machine_costs = {}
    for m, d in opts.items():
        est_load = _estimate_machine_load(ready_ops, m, weights)
        # Weight the estimated load by a contention factor
        load_factor = 1.0 + weights.get('load_weight', 0.3) * est_load
        cost = float(d) * load_factor
        machine_costs[m] = cost
    
    if not machine_costs:
        return 0.0, None, None
    
    # Sort machines by cost
    sorted_machines = sorted(machine_costs.items(), key=lambda x: x[1])
    best_m = sorted_machines[0][0]
    best_cost = sorted_machines[0][1]
    
    if len(sorted_machines) >= 2:
        second_cost = sorted_machines[1][1]
    else:
        second_cost = float('inf')
    
    # Regret is the difference: higher regret means more important to decide now
    regret = second_cost - best_cost
    if regret < 0:
        regret = 0.0
    
    return regret, best_m, best_cost

def dispatch(ready_operations, time):
    """
    Dispatch rule using regret-based look-ahead heuristic.
    
    Selects the operation with the maximum regret (difference between
    best and second-best machine assignment costs), prioritizing decisions
    where machine choice has the highest potential impact on the schedule.
    """
    if not ready_operations:
        return 0
    
    if _elapsed() > _DEADLINE:
        # Safety fallback: Shortest Processing Time
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
    
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0
    
    # Weights for the regret calculation
    weights = {'load_weight': 0.3, 'urgency_weight': 0.5}
    
    # Precompute machine usage counts for contention awareness
    machine_usage = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_usage[m] = machine_usage.get(m, 0) + 1
    
    best_regret = -float('inf')
    best_index = 0
    best_machine = None
    best_combined_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 0.0))
        
        regret, best_m, best_cost = _compute_regret(op, ready_operations, weights)
        
        if best_m is None:
            continue
        
        # Combine regret with urgency: operations with more remaining work
        # are more urgent and should be prioritized
        urgency = weights['urgency_weight'] * (rw + 0.1 * ro)
        
        # The combined score balances regret (importance of machine choice)
        # with urgency (how much work is left)
        # Higher regret means we should decide sooner to avoid suboptimal assignment
        # Higher urgency means the operation is more critical
        combined = regret + urgency / (best_cost + 1e-9)
        
        if combined > best_combined_score:
            best_combined_score = combined
            best_index = i
            best_machine = best_m
    
    if best_machine is None:
        # Fallback: pick the operation with shortest minimum duration
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
        best_index = best_i
        op = ready_operations[best_index]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: opts[k])
        else:
            return best_index
    
    return (best_index, best_machine)
