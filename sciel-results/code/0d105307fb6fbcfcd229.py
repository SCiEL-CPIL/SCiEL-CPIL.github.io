"""Model-and-optimize dispatch for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from scipy.optimize import linear_sum_assignment

_START_TIME = _time.time()
_HARD_LIMIT = 145.0  # seconds - safety margin before 150s kill
_SOFT_MARGIN = 5.0   # seconds buffer

def _time_remaining():
    return _HARD_LIMIT - (_time.time() - _START_TIME)

def _deadline_exceeded():
    return _time_remaining() < _SOFT_MARGIN

_param_cache = {}
_best_params_global = None

def _soft_min(values, beta=3.0):
    """Smooth minimum via log-sum-exp trick: -log(sum(exp(-beta*x)))/beta"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    min_val = float(np.min(arr))
    shifted = arr - min_val
    log_sum = float(np.log(np.sum(np.exp(-beta * shifted))))
    return min_val - log_sum / beta

def _soft_max(values, beta=3.0):
    """Smooth maximum via log-sum-exp trick: log(sum(exp(beta*x)))/beta"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    max_val = float(np.max(arr))
    shifted = arr - max_val
    log_sum = float(np.log(np.sum(np.exp(beta * shifted))))
    return max_val + log_sum / beta

def _build_cost_matrix(ready_operations, time, beta=3.0):
    """
    Build cost matrix for Hungarian assignment with epigraph formulation.
    cost[i][j] = soft_max of completion time if op i runs on machine j,
    incorporating load balancing via auxiliary variable t surrogate.
    """
    n = len(ready_operations)
    if n == 0:
        return None, None
    
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    machines = sorted(all_machines)
    m = len(machines)
    
    if m == 0:
        return None, None
    
    INF = 1e9
    cost_matrix = np.full((n, m), INF, dtype=np.float64)
    
    # Estimate current load per machine from ready operations
    machine_load = np.zeros(m)
    for i, op in enumerate(ready_operations):
        for j, mach in enumerate(machines):
            if mach in op["machine_options"]:
                machine_load[j] += op["machine_options"][mach]
    
    # Normalize loads
    max_load = np.max(machine_load) if np.max(machine_load) > 0 else 1.0
    norm_load = machine_load / max_load
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        for j, mach in enumerate(machines):
            if mach in machine_options:
                dur = machine_options[mach]
                # Completion time if this op starts now on machine j
                comp_time = time + dur
                # Load penalty: prefer less-loaded machines (epigraph t surrogate)
                # The soft-max of completion times across all ops assigned to j
                # approximates the makespan contribution
                load_penalty = norm_load[j] * dur * 0.3
                # Urgency: ops with more remaining work should be prioritized
                urgency_boost = rem_work / (dur + 1.0) * 0.1
                cost_matrix[i][j] = comp_time + load_penalty - urgency_boost
    
    return cost_matrix, machines

def _hungarian_dispatch(ready_operations, time):
    """
    Use Hungarian algorithm with epigraph-based cost to minimize makespan proxy.
    Returns the index of the operation to start next.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    cost_matrix, machines = _build_cost_matrix(ready_operations, time, beta=3.0)
    if cost_matrix is None:
        rem_works = [op["remaining_work"] for op in ready_operations]
        return int(np.argmax(rem_works))
    
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except Exception:
        rem_works = [op["remaining_work"] for op in ready_operations]
        return int(np.argmax(rem_works))
    
    # Assignment: op_index -> machine_index
    assignment = {}
    for r, c in zip(row_ind, col_ind):
        assignment[r] = c
    
    # Score each operation for dispatch priority:
    # Use the assigned machine's completion time and remaining work urgency
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if i in assignment:
            assigned_j = assignment[i]
            assigned_mach = machines[assigned_j]
            assigned_dur = machine_options.get(assigned_mach, 0.0)
        else:
            assigned_dur = _soft_min(list(machine_options.values()), beta=3.0) if machine_options else 0.0
        
        # Criticality: how much this op contributes to the critical path
        # Ops with high remaining work and short duration unblock quickly
        flexibility = len(machine_options)
        urgency = rem_work / (rem_ops + 1.0) if rem_ops > 0 else rem_work
        
        # The operation that, when started, most reduces the makespan
        # is one with high urgency on a less-loaded machine
        scores[i] = urgency * (1.0 / (assigned_dur + 0.5)) * (1.0 + 0.05 * flexibility)
    
    return int(np.argmax(scores))

def _score_operation(op, time, params):
    """Score a single operation using learned parameters."""
    w_mrw, w_mor, w_min_dur, w_constraint, w_ratio, w_critical = params
    
    rem_work = op["remaining_work"]
    rem_ops = op["remaining_ops"]
    machine_options = op["machine_options"]
    
    if machine_options:
        durations = list(machine_options.values())
        sm_dur = _soft_min(durations, beta=3.0)
        avg_dur = float(np.mean(durations))
        max_dur = float(np.max(durations))
        num_machines = len(durations)
    else:
        sm_dur = 0.0
        avg_dur = 0.0
        max_dur = 0.0
        num_machines = 1
    
    f_mrw = rem_work
    f_mor = float(rem_ops) if rem_ops > 0 else 0.0
    f_min_dur = 1.0 / (sm_dur + 1.0)
    f_constraint = 1.0 / (num_machines + 1.0)
    f_ratio = rem_work / (avg_dur + 1.0)
    f_critical = rem_work / (max_dur + 1.0)
    
    score = (
        w_mrw * f_mrw +
        w_mor * f_mor +
        w_min_dur * f_min_dur * 10 +
        w_constraint * f_constraint * 5 +
        w_ratio * f_ratio +
        w_critical * f_critical
    )
    return score

def _simple_dispatch(ready_operations, time, params=None):
    """Fast dispatch using given params or default."""
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    if params is None or _deadline_exceeded():
        rem_works = [op["remaining_work"] for op in ready_operations]
        return int(np.argmax(rem_works))
    
    scores = np.array([_score_operation(op, time, params) for op in ready_operations])
    return int(np.argmax(scores))

def _quick_lookahead(ready_operations, time, params, horizon=3):
    """
    Multi-step greedy simulation to estimate makespan.
    Uses soft-max surrogate for the makespan (auxiliary variable t).
    """
    if _deadline_exceeded():
        return float('inf')
    
    n = len(ready_operations)
    if n == 0:
        return 0.0
    
    # Track state for simulation
    # Build a simplified model: track remaining work per job and per machine
    jobs_remaining = {}
    machines_remaining = {}
    machine_durations = {}  # (job, machine) -> duration
    
    for op in ready_operations:
        job = op["job"]
        jobs_remaining[job] = op["remaining_work"]
        for m, dur in op["machine_options"].items():
            if job not in machines_remaining:
                machines_remaining[job] = {}
            machines_remaining[job][m] = dur
            machine_durations[(job, m)] = dur
    
    # Simulate: greedily dispatch for `horizon` steps
    current_time = time
    remaining_ops = list(ready_operations)
    
    for step in range(horizon):
        if not remaining_ops or _deadline_exceeded():
            break
        
        # Score all remaining ops
        scores = [_score_operation(op, current_time, params) for op in remaining_ops]
        best_idx = int(np.argmax(scores))
        chosen_op = remaining_ops[best_idx]
        
        # Pick the machine with shortest duration for this op
        if chosen_op["machine_options"]:
            best_mach = min(chosen_op["machine_options"], key=chosen_op["machine_options"].get)
            dur = chosen_op["machine_options"][best_mach]
        else:
            dur = 0.0
        
        current_time += dur
        remaining_ops.pop(best_idx)
        
        # Update remaining work for the job
        job = chosen_op["job"]
        if job in jobs_remaining:
            jobs_remaining[job] = max(0, jobs_remaining[job] - dur)
    
    # Estimate final makespan using soft-max surrogate
    # Lower bound: max of remaining job work and remaining machine work
    job_works = list(jobs_remaining.values()) if jobs_remaining else [0]
    # Estimate machine loads from remaining ops
    machine_works = []
    for op in remaining_ops:
        for m, dur in op["machine_options"].items():
            machine_works.append(dur)
    
    if machine_works:
        # Group by machine
        mach_totals = {}
        for op in remaining_ops:
            for m, dur in op["machine_options"].items():
                mach_totals[m] = mach_totals.get(m, 0) + dur
        machine_works = list(mach_totals.values())
    
    all_works = job_works + machine_works + [current_time]
    if all_works:
        makespan_est = _soft_max(all_works, beta=5.0)
    else:
        makespan_est = current_time
    
    return makespan_est

def _evaluate_params(ready_operations, time, params):
    """Evaluate parameters via quick lookahead simulation."""
    if _deadline_exceeded():
        return float('inf')
    return _quick_lookahead(ready_operations, time, params, horizon=3)

def _optimize_params_for_instance(ready_operations, time):
    """
    Optimize dispatch parameters using random search + local refinement.
    Uses soft-max surrogate for makespan evaluation.
    """
    if _deadline_exceeded():
        return None
    
    np.random.seed(42)
    
    best_params = None
    best_score = float('inf')
    
    max_trials = 30
    if _time_remaining() < 30:
        max_trials = 8
    
    default_params = (1.5, 0.5, 0.3, 0.5, 0.8, 0.5)
    
    default_score = _evaluate_params(ready_operations, time, default_params)
    if default_score < best_score:
        best_score = default_score
        best_params = default_params
    
    for _ in range(max_trials):
        if _deadline_exceeded():
            break
        params = tuple(np.random.uniform(0.1, 3.0, size=6))
        score = _evaluate_params(ready_operations, time, params)
        if score < best_score:
            best_score = score
            best_params = params
    
    # Local refinement with tighter noise
    if best_params is not None and _time_remaining() > 15:
        for _ in range(12):
            if _deadline_exceeded():
                break
            noise = np.random.normal(0, 0.2, size=6)
            params = tuple(max(0.05, p + n) for p, n in zip(best_params, noise))
            score = _evaluate_params(ready_operations, time, params)
            if score < best_score:
                best_score = score
                best_params = params
    
    return best_params

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses Hungarian assignment with epigraph-based makespan surrogate.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Deadline guard
    if _deadline_exceeded():
        rem_works = [op["remaining_work"] for op in ready_operations]
        return int(np.argmax(rem_works))
    
    # Primary: Hungarian-based dispatch with epigraph cost
    try:
        result = _hungarian_dispatch(ready_operations, time)
        if 0 <= result < n:
            return result
    except Exception:
        pass
    
    # Fallback: parameter-based dispatch
    # Create a cache key based on the structural state
    jobs_in_state = tuple(sorted(set(op["job"] for op in ready_operations)))
    time_bucket = int(time * 10)  # coarse time bucket for caching
    
    cache_key = (jobs_in_state, time_bucket, n)
    
    if cache_key in _param_cache:
        params = _param_cache[cache_key]
    else:
        if _time_remaining() > 35:
            params = _optimize_params_for_instance(ready_operations, time)
            if params is not None:
                _param_cache[cache_key] = params
                global _best_params_global
                if _best_params_global is None:
                    _best_params_global = params
            else:
                params = _best_params_global if _best_params_global else (1.5, 0.5, 0.3, 0.5, 0.8, 0.5)
        else:
            params = _best_params_global if _best_params_global else (1.5, 0.5, 0.3, 0.5, 0.8, 0.5)
    
    return _simple_dispatch(ready_operations, time, params)
# EVOLVE-BLOCK-END
