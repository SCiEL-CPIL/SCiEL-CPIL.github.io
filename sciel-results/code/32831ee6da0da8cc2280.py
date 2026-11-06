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


def _soft_second_best(costs, temperature=0.5):
    """
    Smooth approximation of the second-best (i.e., second smallest) cost.
    Uses a log-sum-exp style trick: 
      second_best ≈ -1/β * log(Σ_j exp(-β*c_j)) + best
    More precisely, we compute a soft-min over all costs except the hard min,
    but a simpler and stable approach: compute soft-min of all costs, then
    the "regret" is soft_min_all - hard_min. This is a smooth surrogate for
    (second_best - best) when costs are close.
    
    Actually, a cleaner approach: 
      Let c_min = min(costs)
      Let c_soft = -1/β * log(Σ exp(-β*c))  (soft minimum)
      Then regret_smooth = c_soft - c_min  (this is 0 when all costs equal,
      and approaches (second_min - min) when β is large)
    
    For β large enough, this approximates the true regret well.
    """
    if len(costs) < 2:
        return 0.0
    costs_arr = np.array(costs, dtype=float)
    c_min = np.min(costs_arr)
    # Soft minimum with temperature
    # c_soft = -1/β * log(Σ exp(-β*c))
    # To avoid overflow, shift by c_min:
    shifted = costs_arr - c_min
    # exp(-β * shifted) is safe since shifted >= 0
    beta = 1.0 / temperature
    exp_vals = np.exp(-beta * shifted)
    c_soft = c_min - (1.0 / beta) * np.log(np.sum(exp_vals))
    regret = c_soft - c_min
    # Clamp to non-negative
    if regret < 0:
        regret = 0.0
    return regret


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Strategy: Regret-based heuristic with smooth surrogate and load balancing.
    1. For each ready operation, compute machine assignment costs that account
       for duration, machine contention, and estimated future load.
    2. Compute a smooth regret score (difference between soft-min and hard-min
       cost) to identify operations where the machine choice matters most.
    3. Select the operation with highest regret; break ties by urgency
       (remaining work + remaining ops).
    4. Assign the selected operation to its best-cost machine.
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
    
    # Precompute machine contention and load estimates
    machine_count = {}
    machine_load_estimate = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_count[m] = machine_count.get(m, 0) + 1
            machine_load_estimate[m] = machine_load_estimate.get(m, 0.0) + float(d)
    
    # Precompute per-operation urgency
    urgency_scores = np.empty(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 0.0))
        urgency_scores[i] = rw + 0.1 * ro
    
    # Weight parameters
    CONTENT_WEIGHT = 0.3
    FUTURE_LOAD_WEIGHT = 0.2
    SOFT_TEMPERATURE = 0.5  # lower = sharper approximation to true regret
    
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
            future_load = machine_load_estimate.get(m, 0.0) / max(1, len(machine_count))
            
            cost = est_completion + FUTURE_LOAD_WEIGHT * future_load
            machine_costs.append((cost, m))
        
        if len(machine_costs) == 0:
            continue
        
        # Sort by cost to find best machine
        machine_costs.sort(key=lambda x: x[0])
        best_cost, best_m = machine_costs[0]
        
        # Compute smooth regret
        if len(machine_costs) >= 2:
            all_costs = [c for c, _ in machine_costs]
            regret = _soft_second_best(all_costs, SOFT_TEMPERATURE)
        else:
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
