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

def _soft_min(values, temperature=1.0):
    """Soft-min via log-sum-exp for smooth approximation of min."""
    if not values:
        return 0.0
    min_v = min(values)
    if temperature <= 0:
        return min_v
    # Soft-min = -log(sum(exp(-v/T)))/T
    scaled = [-v / temperature for v in values]
    max_scaled = max(scaled)
    log_sum = math.log(sum(math.exp(s - max_scaled) for s in scaled)) + max_scaled
    return -log_sum * temperature

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
    
    # 1. Calculate normalization constants
    max_rw = 0
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
    if max_rw <= 0:
        max_rw = 1.0

    # 2. Estimate machine loads using a weighted approach
    # Consider all machine options, weighted by how "attractive" each machine is
    machine_load = {}
    
    # First pass: collect all machines and their total potential load
    for op in ready_operations:
        mo = op.get("machine_options", {})
        if not mo:
            continue
        # For each machine option, add a fraction of the duration to that machine's load
        # Weight by 1/duration so that shorter-duration machines get more load consideration
        total_weight = sum(1.0 / d for d in mo.values())
        for m, d in mo.items():
            weight = (1.0 / d) / total_weight
            machine_load[m] = machine_load.get(m, 0) + d * weight
            
    if not machine_load:
        # Fallback if no machine options provided (shouldn't happen in valid FJSP)
        return _fast_dispatch(ready_operations, time)

    avg_load = sum(machine_load.values()) / len(machine_load)
    if avg_load <= 0:
        avg_load = 1.0

    # Weights - tuned for better makespan performance
    w_rw = 0.50
    w_spt = 0.25
    w_load = 0.25
    
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
                cost = d * (1.0 + load_factor)
                effective_costs.append(cost)
            
            # Use soft-min to get a smooth approximation of the best machine
            min_effective_cost = _soft_min(effective_costs, temperature=0.5)
            
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
