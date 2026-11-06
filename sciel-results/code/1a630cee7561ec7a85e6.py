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
    # Soft-min = -soft-max(-v)
    neg_v = -v
    max_neg = np.max(neg_v)
    # If max_neg is very negative, shift to avoid underflow
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
    
    Uses a combined scoring approach with soft-min smoothing:
    - Urgency: remaining_ops (more remaining ops = more critical)
    - Speed: soft-min of durations (smooth surrogate for min duration)
    - Contention avoidance: prefer machines with lower total load
    - Critical path slack: remaining_work / soft_min_duration
    - Secondary: lower remaining_work (job closer to completion)
    
    The soft-min provides a smooth surrogate that accounts for all machine
    options, not just the single best one, leading to better load balancing.
    """
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)
    
    # First pass: compute machine loads (sum of durations per machine)
    # to estimate contention
    machine_loads = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for m, dur in machine_options.items():
            if m not in machine_loads:
                machine_loads[m] = 0.0
            machine_loads[m] += dur

    # Compute average load per machine for normalization
    if machine_loads:
        avg_load = sum(machine_loads.values()) / len(machine_loads)
    else:
        avg_load = 1.0

    # Temperature for soft-min: lower = sharper (closer to true min)
    # Higher = smoother (considers more machines)
    soft_min_temp = 0.5

    best_index = 0
    best_score = -float('inf')
    best_secondary = float('inf')

    epsilon = 1e-9

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            sm_duration = 1.0
            contention_penalty = 0.0
            flexibility = 0.0
        else:
            durations = list(machine_options.values())
            if not durations:
                sm_duration = 1.0
                contention_penalty = 0.0
                flexibility = 0.0
            else:
                # Use soft-min as smooth surrogate for min duration
                sm_duration = _soft_min(durations, temperature=soft_min_temp)
                if sm_duration <= 0:
                    sm_duration = 1e-9
                
                # Compute weighted average load using soft-min weights
                # Machines with shorter durations get higher weight
                durations_arr = np.array(durations, dtype=np.float64)
                # Weights: inverse of duration (faster machines matter more)
                # But use soft-min-inspired weighting for smoothness
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
                
                # Weighted average machine load
                loads_arr = np.array([machine_loads.get(m, 0.0) for m in machine_options.keys()], dtype=np.float64)
                weighted_avg_load = float(np.dot(weights, loads_arr))
                
                # Also compute the load on the true minimum-duration machine
                min_duration = float(np.min(durations_arr))
                min_machine_load = float('inf')
                for m, d in machine_options.items():
                    if abs(d - min_duration) < 1e-9:
                        ml = machine_loads.get(m, 0.0)
                        if ml < min_machine_load:
                            min_machine_load = ml
                
                # Blend: use weighted average load as primary contention signal
                # This is smoother than just looking at the min machine
                if avg_load > 0:
                    contention_ratio = weighted_avg_load / avg_load
                else:
                    contention_ratio = 1.0
                
                # Penalize high contention (ratio > 1 means above average load)
                # Use a softer penalty function
                contention_penalty = max(0.0, contention_ratio - 1.0) ** 1.5 * 0.4
                
                # Flexibility: number of machines with near-min duration
                # More flexibility = less risk
                if min_duration > 0:
                    threshold = min_duration * 1.2  # within 20% of min
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

        # Primary score components:
        # 1. Urgency: remaining_ops (more ops left = more critical)
        # 2. Speed: 1/sm_duration (faster completion)
        # 3. Critical path slack: remaining_work / sm_duration (lower = more urgent)
        
        # Base: urgency × speed
        base_score = remaining_ops / sm_duration
        
        # Critical path slack bonus: jobs with low slack (tight) should be prioritized
        # slack = remaining_work / sm_duration; lower slack = more urgent
        if sm_duration > 1e-9:
            slack = remaining_work / sm_duration
            # Penalize high slack (lots of time left relative to processing)
            slack_penalty = min(slack / 10.0, 1.0) * 0.2
        else:
            slack_penalty = 0.0
        
        # Apply contention penalty as multiplicative discount
        # Flexibility bonus: more flexible ops can be scheduled more freely
        flexibility_bonus = flexibility * 0.1
        
        score = base_score * (1.0 - contention_penalty - slack_penalty + flexibility_bonus)

        # Secondary: prefer lower remaining_work (job closer to done)
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
