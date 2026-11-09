"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0

def _fast_dispatch(ready_operations, time):
    """Extremely fast fallback: Longest Remaining Work (LRW) / MWR."""
    if not ready_operations:
        return 0
    max_rw = -1
    best_idx = 0
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
            best_idx = i
    return best_idx

def _soft_min(values, temperature=0.2):
    """Compute soft minimum using log-sum-exp trick for stability."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    # soft_min(x) = -log(sum(exp(-x_i / T))) * T
    neg_vals = [-v for v in values]
    max_neg = max(neg_vals)
    
    # Overflow protection
    if max_neg > 500:
        return min(values)
    
    exp_sum = sum(math.exp((nv - max_neg) / temperature) for nv in neg_vals)
    soft_min_val = -(max_neg + math.log(exp_sum)) * temperature
    
    return max(0.01, soft_min_val)

def _estimate_machine_loads(ready_operations):
    """
    Estimate machine loads by simulating a greedy SPT assignment.
    This provides a better proxy for future congestion than static weights.
    """
    machine_load = {}
    
    # Sort operations by min duration (SPT heuristic for load estimation)
    ops_sorted = []
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if mo:
            min_d = min(mo.values())
            ops_sorted.append((min_d, op))
    
    ops_sorted.sort(key=lambda x: x[0])
    
    # Greedily assign each op to the machine with the least current load
    # among its options, breaking ties by duration
    for min_d, op in ops_sorted:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        
        # Find machine with min (load + duration)
        best_machine = None
        best_cost = float('inf')
        for m, d in mo.items():
            load = machine_load.get(m, 0)
            cost = load + d
            if cost < best_cost:
                best_cost = cost
                best_machine = m
        
        if best_machine is not None:
            d = mo[best_machine]
            machine_load[best_machine] = machine_load.get(best_machine, 0) + d
    
    return machine_load

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 10.0:
        return _fast_dispatch(ready_operations, time)
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0
    
    # 1. Calculate normalization constants
    max_rw = 0
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
    if max_rw <= 0:
        max_rw = 1.0

    # 2. Estimate machine loads using greedy SPT simulation
    machine_load = _estimate_machine_loads(ready_operations)
    
    if not machine_load:
        # Fallback if no machine options provided
        return _fast_dispatch(ready_operations, time)

    avg_load = sum(machine_load.values()) / len(machine_load)
    if avg_load <= 0:
        avg_load = 1.0

    # Weights - Tuned to favor SPT (efficiency) over load balancing
    # Based on observation that reducing SPT weight worsened performance
    w_rw = 0.30
    w_spt = 0.50
    w_load = 0.20
    
    best_score = -float('inf')
    best_idx = 0

    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if not mo:
            min_d = 1.0
            min_effective_cost = 1.0
        else:
            durations = list(mo.values())
            min_d = min(durations)
            
            # Calculate effective cost for each machine using soft-min
            # Cost = Duration * (1 + Load / AvgLoad)
            effective_costs = []
            for m, d in mo.items():
                load = machine_load.get(m, 0)
                load_factor = load / avg_load
                # Penalize high load more heavily
                cost = d * (1.0 + 1.5 * load_factor)
                effective_costs.append(cost)
            
            # Use soft-min to get a smooth approximation of the best machine
            # Lower temperature makes the selection more greedy/sharp
            min_effective_cost = _soft_min(effective_costs, temperature=0.2)
            
            # SPT component: shorter is better
            norm_spt = 1.0 / (min_d + 1.0)
            
            # Load component: lower effective cost is better
            scaled_cost = min_effective_cost / avg_load
            norm_load = 1.0 / (1.0 + scaled_cost)

        # RW component
        norm_rw = rw / max_rw
        
        # Combined Score
        score = (w_rw * norm_rw + 
                 w_spt * norm_spt + 
                 w_load * norm_load)
        
        if score > best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie breaker: prefer shorter processing time
            prev_op = ready_operations[best_idx]
            prev_mo = prev_op.get("machine_options", {})
            prev_min_d = min(prev_mo.values()) if prev_mo else 1.0
            
            cur_min_d = min(mo.values()) if mo else 1.0
            
            if cur_min_d < prev_min_d:
                best_idx = i

    return best_idx

# EVOLVE-BLOCK-END
