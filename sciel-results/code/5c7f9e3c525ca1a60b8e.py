import numpy as np
import time as _time_module

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def _soft_min(values, temperature=1.0):
    """Compute soft-min using log-sum-exp trick for numerical stability."""
    if len(values) == 0:
        return 0.0
    v = np.array(values, dtype=np.float64)
    if len(v) == 1:
        return float(v[0])
    neg_v = -v
    max_neg = np.max(neg_v)
    shifted = neg_v - max_neg
    exp_vals = np.exp(shifted / temperature)
    sum_exp = np.sum(exp_vals)
    if sum_exp < 1e-300:
        return float(v[0])
    log_sum_exp = max_neg + temperature * np.log(sum_exp)
    soft_min_val = -log_sum_exp
    return float(soft_min_val)


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling (makespan minimization).
    
    Fluid-flow network model:
    - Estimate each machine's queue depth as the sum of soft-min durations
      of all ready operations that could use it
    - Dispatch the operation that maximizes the ratio of its job's remaining
      critical path work to the estimated load on its preferred machine
    - This prioritizes operations on less congested machines to prevent bottlenecks
    """
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)
    
    # First pass: compute machine queue depths using soft-min durations
    # Each operation contributes its soft-min duration to each machine it can use
    machine_queue_depths = {}
    
    soft_min_temp = 0.5
    
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
        
        durations = list(machine_options.values())
        if not durations:
            continue
        
        # Compute soft-min duration for this operation
        sm_duration = _soft_min(durations, temperature=soft_min_temp)
        if sm_duration <= 0:
            sm_duration = 1e-9
        
        # Compute weights for distributing load across machines
        # Machines with shorter durations get higher weight (they're more likely to be chosen)
        durations_arr = np.array(durations, dtype=np.float64)
        neg_durs = -durations_arr
        max_neg = np.max(neg_durs)
        shifted = neg_durs - max_neg
        weights = np.exp(shifted / soft_min_temp)
        weight_sum = np.sum(weights)
        if weight_sum < 1e-300:
            weights = np.ones(len(durations_arr))
            weight_sum = len(durations_arr)
        else:
            weights = weights / weight_sum
        
        # Distribute the operation's processing time across machines
        # Each machine gets a share proportional to its weight
        for j, (m, w) in enumerate(zip(machine_options.keys(), weights)):
            if m not in machine_queue_depths:
                machine_queue_depths[m] = 0.0
            machine_queue_depths[m] += sm_duration * w

    # Compute average queue depth for normalization
    if machine_queue_depths:
        avg_queue_depth = sum(machine_queue_depths.values()) / len(machine_queue_depths)
    else:
        avg_queue_depth = 1.0

    best_index = 0
    best_score = -float('inf')
    best_secondary = float('inf')

    epsilon = 1e-9

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            # Fallback: no machine options
            effective_load = 1.0
            flexibility = 0.0
            sm_duration = 1.0
        else:
            durations = list(machine_options.values())
            if not durations:
                effective_load = 1.0
                flexibility = 0.0
                sm_duration = 1.0
            else:
                # Compute soft-min duration
                sm_duration = _soft_min(durations, temperature=soft_min_temp)
                if sm_duration <= 0:
                    sm_duration = 1e-9
                
                # Compute weights for machine preference
                durations_arr = np.array(durations, dtype=np.float64)
                neg_durs = -durations_arr
                max_neg = np.max(neg_durs)
                shifted = neg_durs - max_neg
                weights = np.exp(shifted / soft_min_temp)
                weight_sum = np.sum(weights)
                if weight_sum < 1e-300:
                    weights = np.ones(len(durations_arr))
                    weight_sum = len(durations_arr)
                else:
                    weights = weights / weight_sum
                
                # Effective load: weighted average of queue depths on candidate machines
                loads_arr = np.array([machine_queue_depths.get(m, 0.0) for m in machine_options.keys()], dtype=np.float64)
                effective_load = float(np.dot(weights, loads_arr))
                
                # Also consider the minimum queue depth (best case)
                min_queue_depth = float(np.min(loads_arr)) if len(loads_arr) > 0 else 0.0
                
                # Blend: use 70% weighted average + 30% minimum for robustness
                effective_load = 0.7 * effective_load + 0.3 * min_queue_depth
                
                # Flexibility: number of machines with near-min duration
                min_duration = float(np.min(durations_arr))
                if min_duration > 0:
                    threshold = min_duration * 1.2
                    flexible_count = sum(1 for d in durations if d <= threshold)
                    flexibility = flexible_count / len(durations)
                else:
                    flexibility = 0.0

        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None:
            remaining_work = 0.0

        # Primary score: ratio of remaining critical path work to effective machine load
        # Higher ratio = more urgent (more work relative to available capacity)
        effective_load_safe = max(effective_load, 0.01)
        
        # Critical path ratio: remaining work per unit of machine capacity
        critical_ratio = remaining_work / (sm_duration * effective_load_safe)
        
        # Urgency boost: jobs with more remaining ops are more critical
        urgency_factor = 1.0 + 0.1 * (remaining_ops - 1)
        
        # Flexibility bonus: more flexible operations can be rescheduled more easily
        flexibility_bonus = 1.0 + 0.15 * flexibility
        
        # Congestion penalty: penalize if the effective load is significantly above average
        if avg_queue_depth > 0:
            congestion_ratio = effective_load / avg_queue_depth
            congestion_penalty = max(0.0, congestion_ratio - 1.0) ** 1.5 * 0.3
        else:
            congestion_penalty = 0.0
        
        # Combined score
        score = critical_ratio * urgency_factor * flexibility_bonus * (1.0 - congestion_penalty)

        # Secondary: prefer lower remaining_work (job closer to completion)
        secondary = remaining_work

        if score > best_score + epsilon:
            best_score = score
            best_secondary = secondary
            best_index = i
        elif abs(score - best_score) <= epsilon:
            if secondary < best_secondary:
                best_secondary = secondary
                best_index = i

    return int(best_index)
