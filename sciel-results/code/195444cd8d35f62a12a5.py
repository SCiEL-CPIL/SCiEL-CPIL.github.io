"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a Dynamic Bottleneck Optimization Heuristic:
    1. Estimate the cumulative load on each machine based on the sum of minimum 
       durations of all ready operations eligible for that machine.
    2. Score each operation by the ratio of its job's remaining work to the 
       estimated load of its eligible machines. This prioritizes critical path 
       jobs while avoiding exacerbating machine congestion.
    3. Select the machine for the chosen operation that minimizes a composite 
       cost of duration and load penalty.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-compute features for each operation
    rws = []
    machine_options_list = []
    
    # For machine load estimation
    # Sum of min durations for all ops that can use machine m
    machine_load = {}
    
    # Also track min duration per op for quick access
    min_durs = []
    
    for op in ready_operations:
        mw = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        rws.append(mw)
        machine_options_list.append(machine_options)
        
        if machine_options:
            # Find min duration for this op
            best_dur = min(machine_options.values())
            min_durs.append(best_dur)
            
            # Add this op's min duration to the load of all machines it can use
            # This approximates the total work waiting for each machine
            for m, d in machine_options.items():
                machine_load[m] = machine_load.get(m, 0) + d
        else:
            min_durs.append(0)
            
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Calculate scores based on Dynamic Bottleneck Optimization
    # Score = remaining_work / (1 + estimated_machine_load)
    # We want to maximize this score.
    
    scores = []
    best_machines = []
    
    # Normalization for remaining work to keep scores in a reasonable range
    max_rw = max(rws) if rws else 1
    if max_rw == 0:
        max_rw = 1

    # Weights for the composite machine selection cost
    # cost = duration + lambda * (load / max_load)
    # lambda controls how strongly we penalize busy machines
    LAMBDA = 1.0
    
    # Find max load for normalization
    max_load = max(machine_load.values()) if machine_load else 1
    if max_load == 0:
        max_load = 1

    for i in range(n):
        rw = rws[i]
        mo = machine_options_list[i]
        
        if not mo:
            # Fallback
            best_m = None
            score = 0
        else:
            # Calculate the "bottleneck" score for this operation
            # We estimate the "effective load" for this op as the minimum load
            # among its eligible machines? Or the average?
            # If an op can use Machine A (load 10) or Machine B (load 100),
            # it is effectively "unblocked" by the existence of A.
            # So the "constraint" on this op is the *minimum* load among its options.
            # However, if we pick the op, we will assign it to one machine.
            # Let's use the minimum load among eligible machines as the "bottleneck"
            # that limits this op's progress.
            
            eligible_loads = [machine_load.get(m, 0) for m in mo.keys()]
            min_load = min(eligible_loads) if eligible_loads else 0
            
            # Score: Higher RW is better, Lower Load is better.
            # Ratio: RW / (1 + min_load)
            # This penalizes ops that are stuck on busy machines.
            
            # Normalize RW to [0, 1]
            rw_norm = rw / max_rw
            
            # Normalize Load to [0, 1]
            load_norm = min_load / max_load
            
            # Composite Score
            # We want to maximize rw_norm and minimize load_norm.
            # score = rw_norm * (1 - 0.5 * load_norm)
            # Or simply the ratio:
            score = rw_norm / (1.0 + load_norm)
            
            # Select best machine for this operation
            # Minimize: duration + LAMBDA * (machine_specific_load / max_load)
            best_m = None
            best_cost = float('inf')
            
            for m, d in mo.items():
                m_load = machine_load.get(m, 0)
                load_penalty = LAMBDA * (m_load / max_load)
                cost = d + load_penalty
                
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
            
            best_machines.append(best_m)
        
        scores.append(score)
        
    # Find best index
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
            
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
