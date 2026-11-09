"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a Dynamic Bottleneck Mitigation heuristic combined with State-Dependent
    Composite Scoring:
    1. Model: Estimate the "phase" of the schedule based on remaining work distribution.
    2. Optimize: 
       - Machine Selection: Minimize estimated completion time relative to global max load,
         while accounting for immediate contention (queue depth proxy).
       - Operation Selection: Balance Long Remaining Processing Time (LRPT) for critical path
         with Short Remaining Processing Time (SRPT) for tail clearing, weighted by phase.
       - Efficiency: Reward operations that move a lot of work quickly on the selected machine.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-compute features for each operation
    rws = []
    min_durs = []
    machine_options_list = []
    remaining_ops_counts = []
    
    # For machine contention analysis
    machine_usage_counts = {}
    machine_min_durs = {}
    
    for op in ready_operations:
        mw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        rws.append(mw)
        remaining_ops_counts.append(ro)
        machine_options_list.append(machine_options)
        
        if machine_options:
            # Find best machine (shortest duration)
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            best_dur = machine_options[best_machine]
            min_durs.append(best_dur)
            
            # Count how many ready ops can run on each machine
            # This is a proxy for machine load/congestion
            for m in machine_options.keys():
                machine_usage_counts[m] = machine_usage_counts.get(m, 0) + 1
                # Track the minimum duration required for this machine across all ops
                d = machine_options[m]
                if m not in machine_min_durs or d < machine_min_durs[m]:
                    machine_min_durs[m] = d
        else:
            min_durs.append(0)
            
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Calculate average remaining work to determine "phase"
    avg_rw = sum(rws) / n
    max_rw = max(rws) if rws else 1
    min_rw = min(rws) if rws else 0
    
    # Phase indicator: ratio of avg to max remaining work
    # Early phase: high ratio (jobs similar, lots of work)
    # Late phase: low ratio (jobs diverged, little work)
    phase_ratio = avg_rw / (max_rw + 1e-9) if max_rw > 0 else 1.0
    
    # Estimate global max machine load to normalize completion times
    # We approximate "load" by summing the minimum durations of ops eligible for that machine
    # This is a static proxy for congestion.
    machine_estimated_load = {}
    for i in range(n):
        mo = machine_options_list[i]
        if mo:
            for m in mo.keys():
                # Use the duration specific to this op on this machine
                machine_estimated_load[m] = machine_estimated_load.get(m, 0) + mo[m]
    
    max_load = max(machine_estimated_load.values()) if machine_estimated_load else 1.0
    if max_load == 0:
        max_load = 1.0

    scores = []
    best_machines = []
    
    # Strategy: Dynamic Bottleneck Mitigation + Phase-Adaptive Efficiency
    # 
    # 1. Machine Selection:
    #    For each op, find the machine that minimizes:
    #    cost = duration + contention_penalty + load_relative_penalty
    #    - duration: actual time to process
    #    - contention_penalty: smooth function of how many other ops want this machine
    #    - load_relative_penalty: (estimated_total_load_on_machine / max_load) * weight
    #      This explicitly routes away from globally congested machines.
    
    # 2. Operation Scoring:
    #    score = (w_eff * efficiency + w_bott * bottleneck_factor) * cont_factor
    #    - efficiency: rw / (selected_duration + 1)
    #    - bottleneck_factor: rw / (total_eligible_load + 1)
    #    - cont_factor: penalty for high contention on selected machine
    #    - w_eff, w_bott: phase-dependent weights
    
    for i in range(n):
        rw = rws[i]
        mo = machine_options_list[i]
        
        if not mo:
            best_m = None
            score = 0.0
        else:
            # Select best machine using composite cost
            best_m = None
            best_cost = float('inf')
            best_cont = 0
            best_dur = float('inf')
            
            for m, d in mo.items():
                cnt = machine_usage_counts.get(m, 0)
                
                # Smooth contention penalty
                # For small cnt, penalty ~ cnt; for large cnt, saturates
                smooth_pen = 0.3 * cnt / (1.0 + 0.5 * cnt)
                
                # Load relative penalty
                # Estimate how busy this machine is relative to the busiest one
                load_est = machine_estimated_load.get(m, 0)
                load_rel = load_est / max_load
                load_pen = 0.2 * load_rel
                
                cost = d + smooth_pen + load_pen
                
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
                    best_cont = cnt
                    best_dur = d
            
            best_machines.append(best_m)
            
            # Soft-min duration for efficiency calculation (using the selected machine's duration)
            # We use the actual duration of the selected machine for a more precise efficiency measure
            # but also consider the "soft min" of all options to capture flexibility
            durs_list = list(mo.values())
            if len(durs_list) == 1:
                soft_min_dur = durs_list[0]
            else:
                # Soft min via log-sum-exp with temperature T=1.0
                T = 1.0
                neg_durs = [-d / T for d in durs_list]
                max_neg = max(neg_durs)
                sum_exp = sum(math.exp(x - max_neg) for x in neg_durs)
                soft_min_dur = -(max_neg + math.log(sum_exp)) * T
                soft_min_dur = max(soft_min_dur, 1e-9)
            
            # Efficiency: remaining work per unit time on the *selected* machine
            # Using best_dur (actual selected) is more relevant for the immediate step
            efficiency = rw / (best_dur + 1.0)
            
            # Bottleneck factor: how "critical" this job is relative to machine capacity
            total_eligible_load = sum(machine_estimated_load.get(m, 0) for m in mo.keys())
            if total_eligible_load > 0:
                bottleneck_factor = rw / (total_eligible_load + 1.0)
            else:
                bottleneck_factor = 1.0
            
            # Contention factor: smooth penalty for the selected machine
            cont_factor = 1.0 / (1.0 + 0.7 * best_cont)
            
            # Phase-adaptive weighting:
            # Early phase: weight efficiency more (keep critical jobs moving)
            # Late phase: weight bottleneck more (clear congested paths)
            w_eff = 0.6 + 0.4 * (1.0 - phase_ratio)
            w_bott = 0.4 + 0.4 * phase_ratio
            
            # Combine scores
            score = (w_eff * efficiency + w_bott * bottleneck_factor) * cont_factor
            
            # Additional bonus for reducing the estimated max load
            # If this op is on the currently most loaded machine, penalize slightly
            # This is implicitly handled by load_pen in machine selection, but
            # let's add a small global load reduction bonus
            # If the selected machine is the most loaded, we might want to avoid it
            # unless the efficiency is very high.
            # The cost function in machine selection already tries to avoid high load.
            
        scores.append(score)
        
    # Find best
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
