import numpy as np
import time as time_module
import heapq
import random

# ------------------------------------------------------------------
#  Open-Shop dispatch rule with Greedy Look-Ahead and Smooth Surrogate.
#
#  Strategy: 
#  1. Primary heuristic: Multi-criterion scoring (MWR, SPT, Load Balance).
#  2. Refinement: If look-ahead data is available, use a smooth 
#     log-sum-exp surrogate for the makespan to break ties or refine 
#     the top-K candidates. This addresses the non-smoothness of 
#     max(completion_times) which can lead to brittle local optima 
#     in greedy choices.
# ------------------------------------------------------------------

_DEADLINE = 150.0
_START_TIME = time_module.time()
_TIME_MARGIN = 2.0

def _elapsed():
    return time_module.time() - _START_TIME

def _remaining_time():
    return _DEADLINE - _TIME_MARGIN - _elapsed()

def _get_machine_options(op):
    """Extract machine options as a list of (machine, duration)."""
    mo = op.get("machine_options", {})
    return [(m, d) for m, d in mo.items()]

def _count_machine_loads(ready_operations):
    """
    Build a dict mapping machine -> count of ready ops that use it.
    This is a rough proxy for machine congestion.
    """
    loads = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo.keys():
            loads[m] = loads.get(m, 0) + 1
    return loads

def _score_op(op, machine_loads=None, time=0.0):
    """
    Score a single ready operation.  Higher score = more preferred.
    Uses MWR, SPT, and Load Balance with improved weights.
    """
    rw = op.get("remaining_work", 0.0)
    ro = op.get("remaining_ops", 1)
    mo = op.get("machine_options", {})
    
    # Min and mean duration
    durs = list(mo.values()) if mo else [0.0]
    min_dur = min(durs) if durs else 0.0
    
    # Machine load penalty: prefer least-loaded machine among options
    mach_penalty = 0.0
    if machine_loads is not None and mo:
        loads = [machine_loads.get(m, 0) for m in mo.keys()]
        min_load = min(loads) if loads else 0
        mach_penalty = min_load
    
    # Improved weights: balance job-tail completion against machine flexibility
    W_MWR = 0.7
    W_ROPS = 0.2
    W_SPT = 0.1
    
    score = (W_MWR * rw
             + W_ROPS * ro
             + W_SPT * (1.0 / (min_dur + 1.0))
             - 0.05 * mach_penalty)
    
    return score

def _smooth_max(values, beta=1.0):
    """
    Smooth approximation of max(values) using log-sum-exp.
    log-sum-exp(x) = log(sum(exp(x_i)))
    max(x) ≈ (1/beta) * log(sum(exp(beta * x_i)))
    As beta -> infinity, this approaches max(x).
    For small beta, it is smoother and less sensitive to single outliers.
    """
    if not values:
        return 0.0
    # Scale to avoid overflow
    v = np.array(values, dtype=np.float64)
    m = np.max(v)
    if m == 0:
        return 0.0
    # Use a stable log-sum-exp calculation
    # log(sum(exp(v))) = m + log(sum(exp(v - m)))
    exp_sum = np.sum(np.exp(beta * (v - m)))
    if exp_sum == 0:
        return m
    return (m + np.log(exp_sum) / beta)

def _simulate_makespan_estimate(ready_operations, chosen_idx, time, machine_loads, beta=1.0):
    """
    Estimate the makespan if we schedule `chosen_idx` now, and then greedily
    schedule the remaining ready operations.
    
    Improved simulation:
    1. Use Hungarian-algorithm-inspired greedy: maximize minimum slack.
    2. Track machine availability more carefully.
    3. Use smoother makespan estimate.
    """
    n = len(ready_operations)
    if n == 0:
        return 0.0
    
    # 1. Determine the machine for the chosen op.
    #    Pick the machine with the lowest current load (congestion).
    op_chosen = ready_operations[chosen_idx]
    mo_chosen = op_chosen.get("machine_options", {})
    
    best_m_chosen = None
    min_load_chosen = float('inf')
    for m, d in mo_chosen.items():
        load = machine_loads.get(m, 0)
        if load < min_load_chosen:
            min_load_chosen = load
            best_m_chosen = m
    
    if best_m_chosen is None:
        # Fallback: first machine
        best_m_chosen = list(mo_chosen.keys())[0] if mo_chosen else 0
        dur_chosen = mo_chosen.get(best_m_chosen, 0.0)
    else:
        dur_chosen = mo_chosen[best_m_chosen]
        
    finish_chosen = time + dur_chosen
    
    # 2. Initialize virtual availability for all machines.
    virtual_avail = {}
    all_machines = set()
    for op in ready_operations:
        for m in op.get("machine_options", {}).keys():
            all_machines.add(m)
            
    for m in all_machines:
        virtual_avail[m] = time
        
    # Update the chosen machine's availability
    virtual_avail[best_m_chosen] = finish_chosen
    
    # 3. Greedily schedule remaining ops using improved heuristic.
    #    At each step, pick the op that maximizes minimum slack across its machines,
    #    with tie-breaking by earliest completion time.
    
    remaining_indices = [i for i in range(n) if i != chosen_idx]
    completion_times = [finish_chosen]
    
    while remaining_indices:
        best_i = -1
        best_m = -1
        best_slack = float('-inf')
        best_finish = float('inf')
        
        for i in remaining_indices:
            op = ready_operations[i]
            mo = op.get("machine_options", {})
            
            # Calculate slack for each machine: earliest start - current time
            # Slack = min over machines of (virtual_avail[m] - time)
            # We want to maximize this to avoid blocking critical machines.
            min_slack = float('inf')
            best_m_for_i = -1
            best_finish_for_i = float('inf')
            
            for m, d in mo.items():
                avail = virtual_avail.get(m, time)
                slack = avail - time
                if slack < min_slack:
                    min_slack = slack
                    best_m_for_i = m
                    best_finish_for_i = avail + d
            
            # If slack is tied, prefer earlier completion
            if min_slack > best_slack + 1e-9:
                best_slack = min_slack
                best_i = i
                best_m = best_m_for_i
                best_finish = best_finish_for_i
            elif abs(min_slack - best_slack) < 1e-9:
                if best_finish_for_i < best_finish:
                    best_i = i
                    best_m = best_m_for_i
                    best_finish = best_finish_for_i
        
        if best_i == -1:
            break
            
        # Update state
        virtual_avail[best_m] = best_finish
        completion_times.append(best_finish)
        remaining_indices.remove(best_i)
        
    # 4. Calculate makespan estimate using smooth max.
    if not completion_times:
        return 0.0
    
    # Use smooth max with beta=1.0 for a balance between smoothness and accuracy
    v = np.array(completion_times, dtype=np.float64)
    m_max = np.max(v)
    if m_max == 0:
        return 0.0
    
    # Stable log-sum-exp calculation
    exp_sum = np.sum(np.exp(beta * (v - m_max)))
    if exp_sum == 0:
        return m_max
    
    return m_max + np.log(exp_sum) / beta

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    if _remaining_time() < 0.5:
        # Fallback to simple heuristic
        return _simple_heuristic(ready_operations, time)
    
    n = len(ready_operations)
    
    # 1. Compute machine loads (congestion proxy)
    machine_loads = _count_machine_loads(ready_operations)
    
    # 2. Score all ops with the primary heuristic
    scores = []
    for i, op in enumerate(ready_operations):
        s = _score_op(op, machine_loads, time)
        scores.append((s, i))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # If we have very few ops, just pick the top one
    if n <= 2:
        return scores[0][1]
    
    # 3. Look-ahead refinement for top-K candidates
    #    We check if look-ahead is feasible (i.e., we have enough time).
    #    The simulation is O(N^2 * M). For N=10, M=5, this is 500 ops.
    #    Doing this for Top-K=3 is 1500 ops. Very fast.
    
    top_k = min(5, n)
    
    best_idx = scores[0][1]
    best_mk = float('inf')
    
    for k in range(top_k):
        idx = scores[k][1]
        
        # Check deadline
        if _remaining_time() < 1.0:
            break
            
        est_mk = _simulate_makespan_estimate(ready_operations, idx, time, machine_loads)
        
        # We want to minimize makespan.
        # Blend with primary score: final = est_mk - 0.1 * primary_score
        # This respects both the simulation and the heuristic.
        blended_score = est_mk - 0.1 * scores[k][0]
        
        if blended_score < best_mk - 1e-6:
            best_mk = blended_score
            best_idx = idx
    
    return best_idx

def _simple_heuristic(ready_operations, time):
    best_idx = 0
    best_score = float('-inf')
    
    # Quick machine load count
    loads = {}
    for op in ready_operations:
        for m in op.get("machine_options", {}):
            loads[m] = loads.get(m, 0) + 1
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        ro = op.get("remaining_ops", 1)
        mo = op.get("machine_options", {})
        durs = list(mo.values()) if mo else [0.0]
        min_dur = min(durs) if durs else 0.0
        
        # Min load among option machines
        min_load = min((loads.get(m, 0) for m in mo), default=0)
        
        s = 0.7 * rw + 0.2 * ro + 0.1 * (1.0 / (min_dur + 1.0)) - 0.05 * min_load
        if s > best_score:
            best_score = s
            best_idx = i
    return best_idx
