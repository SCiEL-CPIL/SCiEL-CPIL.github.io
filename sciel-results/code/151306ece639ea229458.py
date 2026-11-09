"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill

# Tuning parameters
_T_SOFT_MIN = 2.0  # Lower T makes soft-min closer to hard min
_W_RW = 0.8        # Weight for Remaining Work
_W_DUR = 0.1       # Weight for Duration (efficiency)
_W_CONT = 0.1      # Weight for Contention (congestion)
_LAMBDA_CONT = 0.5 # Penalty for machine contention in machine selection


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a State-Dependent Composite Heuristic (Model-and-Optimize approach):
    1. Model: Estimate the "phase" of the schedule based on the distribution of 
       remaining work. 
    2. Optimize: Adjust heuristic weights dynamically.
       - Early phase (high avg remaining work): Prioritize Long Remaining Processing 
         Time (LRPT) to keep critical path jobs moving.
       - Late phase (low avg remaining work): Prioritize Short Remaining Processing 
         Time (SRPT) to clear the queue and reduce tail effects.
    3. Machine Selection: Select the machine that minimizes a composite cost of 
       duration and estimated contention (number of other ready ops eligible for 
       that machine).
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
    
    # Heuristic for phase:
    # If avg_rw is close to max_rw, jobs are similar (or early).
    # If avg_rw is small, we are likely late (assuming work decreases).
    # We use a sigmoid-like transition to blend weights.
    # Scale: Assume total horizon is roughly 2 * max_rw? No, let's use relative variance.
    
    # Let's use the ratio of avg to max as a phase indicator.
    # Early: jobs have lots of work. Late: jobs have little work.
    # Without a global horizon, we can't be perfect. 
    # Alternative: Use the variance of remaining work.
    
    # Let's stick to a robust weighted sum but with machine contention penalty.
    # This is a known improvement over simple LRPT/SRPT.
    
    max_dur = max(min_durs) if min_durs else 1
    min_dur = min(min_durs) if min_durs else 1
    
    # Calculate Contention Score for each op's best machine
    # High contention = many other ops want this machine.
    # We prefer machines with lower contention.
    
    scores = []
    best_machines = []
    
    # Pre-calculate soft-min values for better normalization
    soft_min_vals = [0.0] * n
    
    for i in range(n):
        mo = machine_options_list[i]
        if not mo:
            continue
            
        costs = []
        for m, d in mo.items():
            cnt = machine_usage_counts.get(m, 0)
            # Use higher penalty to avoid congestion
            cost = d + _LAMBDA_CONT * cnt
            costs.append(cost)
            
        if costs:
            # Soft-min calculation
            # soft_min(x) = -T * log(sum(exp(-x_i / T)))
            # Stabilized version:
            min_cost = min(costs)
            # To avoid underflow in exp, we can shift, but for small N and reasonable T, direct is often okay.
            # However, let's use the stable form:
            # Let m = min(costs).
            # sum(exp(-c/T)) = exp(-m/T) * sum(exp(-(c-m)/T))
            # log(sum(exp(-c/T))) = -m/T + log(sum(exp(-(c-m)/T)))
            # soft_min = -T * (-m/T + log(...)) = m - T * log(sum(exp(-(c-m)/T)))
            
            sum_exp = 0.0
            for c in costs:
                # c - min_cost >= 0, so exp(-(c-min_cost)/T) <= 1. No overflow.
                sum_exp += math.exp(-(c - min_cost) / _T_SOFT_MIN)
            
            soft_min_val = min_cost - _T_SOFT_MIN * math.log(sum_exp)
            soft_min_vals[i] = soft_min_val
    
    # Normalize soft-min values
    if n > 0:
        min_sm = min(soft_min_vals)
        max_sm = max(soft_min_vals)
        range_sm = max_sm - min_sm
    else:
        min_sm = 0
        max_sm = 0
        range_sm = 0
    
    for i in range(n):
        rw = rws[i]
        mo = machine_options_list[i]
        
        if not mo:
            # Fallback
            best_m = None
            contention = 0
            eff_dur_norm = 0.0
        else:
            # Select best machine based on cost
            best_m = None
            best_cost = float('inf')
            for m, d in mo.items():
                cnt = machine_usage_counts.get(m, 0)
                cost = d + _LAMBDA_CONT * cnt
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
            
            contention = machine_usage_counts.get(best_m, 0)
            
            # Use pre-calculated soft-min for duration feature
            if range_sm > 0:
                eff_dur_norm = (soft_min_vals[i] - min_sm) / range_sm
            else:
                eff_dur_norm = 0.0
        
        best_machines.append(best_m)
        
        rw_score = rw / max_rw if max_rw > 0 else 0
        
        # Contention score: lower is better.
        cont_score = contention / n if n > 0 else 0
        
        # Composite Score
        # We want to MAXIMIZE score.
        score = _W_RW * rw_score + _W_DUR * (1 - eff_dur_norm) + _W_CONT * (1 - cont_score)
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
