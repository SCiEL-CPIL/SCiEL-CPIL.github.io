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

# UCB state for machine exploration
_uca_state = {}  # machine -> (visits, total_cost)
_ucb_exploration = 1.0  # exploration parameter

def _ucb_score(machine, current_time, machine_load):
    """Compute UCB score for a machine: exploit (low load) + explore (underutilized)."""
    if machine not in _uca_state:
        _uca_state[machine] = [1, 0.0]  # Initialize with 1 visit, 0 cost
    
    visits, total_cost = _uca_state[machine]
    mean_cost = total_cost / visits
    
    # UCB: mean_cost + sqrt(2 * ln(1+total_visits) / visits)
    # We use total dispatch count as a proxy for total visits
    total_dispatches = sum(v for v, c in _uca_state.values())
    exploration_bonus = math.sqrt(2.0 * math.log(1 + total_dispatches) / visits) * _ucb_exploration
    
    # Lower is better: combine mean load cost with exploration bonus
    # Normalize load to [0,1] range for consistency
    max_possible_load = max(machine_load.values()) if machine_load else 1.0
    normalized_load = machine_load.get(machine, 0) / max_possible_load if max_possible_load > 0 else 0.5
    
    return normalized_load + exploration_bonus

def _update_ucb(machine, cost):
    """Update UCB state after dispatching to a machine."""
    if machine not in _uca_state:
        _uca_state[machine] = [0, 0.0]
    _uca_state[machine][0] += 1
    _uca_state[machine][1] += cost

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
    
    # Precompute normalization constants
    max_rw = max(op.get("remaining_work", 0) for op in ready_operations)
    if max_rw <= 0:
        max_rw = 1.0
    
    # Estimate machine loads by summing durations across ready ops
    # This helps identify bottleneck machines
    machine_load = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m, d in mo.items():
            machine_load[m] = machine_load.get(m, 0) + d
    
    # Average load per machine
    if machine_load:
        avg_load = sum(machine_load.values()) / len(machine_load)
    else:
        avg_load = 1.0
    
    # Compute UCB scores for all machines
    ucb_scores = {}
    for m in machine_load:
        ucb_scores[m] = _ucb_score(m, time, machine_load)
    
    # Weights: critical path (remaining work) is dominant
    w_rw = 0.55
    w_spt = 0.25
    w_load = 0.20
    
    best_score = -float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 1)
        
        if not mo:
            min_d = 1.0
            best_machine_load = avg_load
        else:
            durations = list(mo.values())
            min_d = min(durations)
            
            # Use soft-min to consider flexibility: if multiple machines have similar durations,
            # the effective cost is lower due to flexibility
            soft_min_d = _soft_min(durations, temperature=0.5)
            
            # Use UCB to select machine: balance exploitation (low load) with exploration
            # Score each candidate machine using UCB, prefer lower UCB score
            min_d_threshold = min_d * 1.2  # Allow up to 20% longer if it reduces load
            candidate_machines = [(m, d) for m, d in mo.items() if d <= min_d_threshold]
            
            if candidate_machines:
                # Find machine with lowest UCB score among candidates
                best_ucb = float('inf')
                best_machine = None
                for m, d in candidate_machines:
                    ucb_s = ucb_scores.get(m, 1.0)
                    if ucb_s < best_ucb:
                        best_ucb = ucb_s
                        best_machine = m
                
                if best_machine is not None:
                    best_machine_load = machine_load.get(best_machine, 0)
                    # Update UCB state for the selected machine
                    _update_ucb(best_machine, best_machine_load)
                else:
                    best_machine_load = min(machine_load.get(m, 0) for m, d in candidate_machines)
            else:
                best_machine_load = machine_load.get(list(mo.keys())[0], 0)
        
        # Normalize components
        norm_rw = rw / max_rw
        
        # SPT component: shorter processing time is better
        # Use inverse with offset to avoid division by zero
        norm_spt = 1.0 / (min_d + 1.0)
        
        # Load balancing: prefer operations that can run on less-loaded machines
        # Lower load is better, so invert
        if avg_load > 0:
            norm_load = 1.0 - (best_machine_load / (2.0 * avg_load))
            norm_load = max(0.0, min(1.0, norm_load))
        else:
            norm_load = 0.5
        
        # UCB exploration bonus: small bonus for operations that can use underexplored machines
        if candidate_machines:
            min_ucb = min(ucb_scores.get(m, 1.0) for m, d in candidate_machines)
            max_ucb = max(ucb_scores.get(m, 1.0) for m, d in candidate_machines)
            if max_ucb > min_ucb:
                ucb_bonus = 1.0 - ((min_ucb - min_ucb) / (max_ucb - min_ucb))
            else:
                ucb_bonus = 0.5
        else:
            ucb_bonus = 0.5
        
        # Critical path bonus: operations with fewer remaining steps but high remaining work
        # are on the critical path and should be prioritized
        # This captures the idea that we want to unblock jobs that are close to completion
        # but still have significant work
        if remaining_ops > 0:
            critical_bonus = 1.0 / (1.0 + remaining_ops)
        else:
            critical_bonus = 0.0
        
        # Combined score with UCB exploration component
        w_ucb = 0.08
        score = (w_rw * norm_rw + 
                 w_spt * norm_spt + 
                 w_load * norm_load +
                 0.05 * critical_bonus +
                 w_ucb * ucb_bonus)
        
        if score > best_score:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            # Tie breaker: prefer shorter processing time, then lower load
            prev_op = ready_operations[best_idx]
            prev_mo = prev_op.get("machine_options", {})
            prev_min_d = min(prev_mo.values()) if prev_mo else 1.0
            if min_d < prev_min_d:
                best_idx = i
            elif abs(min_d - prev_min_d) < 1e-9:
                prev_best_load = machine_load.get(list(prev_mo.keys())[0], 0) if prev_mo else 0
                if best_machine_load < prev_best_load:
                    best_idx = i
    
    return best_idx

# EVOLVE-BLOCK-END
