"""Model-and-optimize dispatch for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment

_START_TIME = _time.time()
_HARD_LIMIT = 130.0  # seconds - safety margin before 150s kill
_SOFT_MARGIN = 8.0   # seconds buffer

def _time_remaining():
    return _HARD_LIMIT - (_time.time() - _START_TIME)

def _deadline_exceeded():
    return _time_remaining() < _SOFT_MARGIN

# Cache for instances and optimization results
_instance_cache = {}  # key -> (ready_ops_list, makespan_by_choice)
_param_cache = {}  # key -> best_params
_best_params_global = None
_instance_count = [0]

def _instance_key(ready_operations, time):
    """Create a hashable key for an instance state."""
    # Use the job/op/machine structure to identify the instance
    jobs = set()
    machines = set()
    key_parts = []
    for op in ready_operations:
        jobs.add(op["job"])
        key_parts.append((op["job"], op["op"]))
        for m in op["machine_options"]:
            machines.add(m)
    # Sort for consistency
    key = (tuple(sorted(key_parts)), tuple(sorted(jobs)), tuple(sorted(machines)), round(time, 6))
    return key

def _soft_min(values, beta=1.0):
    """Smooth minimum via log-sum-exp trick: -log(sum(exp(-beta*x)))/beta"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    min_val = np.min(arr)
    # Numerical stability: shift by min
    shifted = arr - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - log_sum / beta

def _soft_max(values, beta=1.0):
    """Smooth maximum via log-sum-exp trick: log(sum(exp(beta*x)))/beta"""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    max_val = np.max(arr)
    shifted = arr - max_val
    log_sum = np.log(np.sum(np.exp(beta * shifted)))
    return max_val + log_sum / beta

def _hungarian_dispatch(ready_operations, time):
    """
    Use Hungarian algorithm on a cost matrix to assign operations to machines,
    minimizing the maximum completion time (makespan proxy).
    
    Cost matrix C[i][j] = estimated completion time if operation i runs on machine j.
    We want to minimize max(C[i][assigned_machine(i)]).
    
    Since Hungarian minimizes sum, we use a transformation:
    - Build cost matrix where cost[i][j] = time + duration[i][j] + load_j
    - The assignment that minimizes the sum tends to balance loads
    - Add a penalty term for the maximum to encourage min-max behavior
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Collect all machines from ready operations
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    machines = sorted(all_machines)
    m = len(machines)
    
    if m == 0:
        return 0
    
    # Build cost matrix: n operations x m machines
    # cost[i][j] = estimated makespan contribution if op i is assigned to machine j
    # If op i cannot run on machine j, set cost to a large value
    
    INF = 1e9
    cost_matrix = np.full((n, m), INF, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        for j, mach in enumerate(machines):
            if mach in machine_options:
                dur = machine_options[mach]
                # Completion time if this op runs on machine j
                # Add a load-balancing penalty: prefer machines with less remaining work
                # Estimate machine load from other ready ops that could use this machine
                cost_matrix[i][j] = dur
    
    # Hungarian algorithm requires square or rectangular matrix
    # If n > m, we need to pad; if n < m, we need to pad rows
    # linear_sum_assignment handles rectangular matrices
    
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except Exception:
        # Fallback to simple heuristic
        rem_works = [op["remaining_work"] for op in ready_operations]
        return int(np.argmax(rem_works))
    
    # The Hungarian algorithm gives us an assignment that minimizes total cost
    # We want to pick the operation that, in this optimal assignment, has the 
    # highest priority (largest remaining work or most constrained)
    
    # Build assignment: op_index -> machine_index
    assignment = {}
    for r, c in zip(row_ind, col_ind):
        assignment[r] = c
    
    # Score each operation based on:
    # 1. Its position in the optimal assignment (which machine it's assigned to)
    # 2. Its remaining work (urgency)
    # 3. The soft-min of its available durations (flexibility)
    
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        durations = list(machine_options.values())
        if durations:
            sm = _soft_min(durations, beta=2.0)
            smax = _soft_max(durations, beta=2.0)
            avg_dur = np.mean(durations)
        else:
            sm = 0.0
            smax = 0.0
            avg_dur = 0.0
        
        # Which machine is this op assigned to in the Hungarian solution?
        if i in assignment:
            assigned_mach = machines[assignment[i]]
            assigned_dur = machine_options.get(assigned_mach, avg_dur)
        else:
            assigned_dur = sm
        
        # Score: prioritize ops with high remaining work, short assigned duration,
        # and high flexibility (many machine options)
        flexibility = len(machine_options)
        urgency = rem_work / (rem_ops + 1.0) if rem_ops > 0 else rem_work
        
        # The Hungarian assignment tells us a good machine for each op.
        # We want to start the op that has the most "critical" role in the assignment.
        # Use: urgency * (1 / assigned_dur) as a proxy for how quickly this op 
        # unblocks the schedule
        
        scores[i] = urgency * (1.0 / (assigned_dur + 0.1)) * (1.0 + 0.1 * flexibility)
    
    return int(np.argmax(scores))

def _simple_dispatch(ready_operations, time, params=None):
    """Fast dispatch using given params or default."""
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    if params is None or _deadline_exceeded():
        # Fast fallback: most remaining work
        rem_works = [op["remaining_work"] for op in ready_operations]
        return int(np.argmax(rem_works))
    
    # Use learned params
    # params = (w_mrw, w_mor, w_min_dur, w_constraint, w_ratio, w_critical)
    w_mrw, w_mor, w_min_dur, w_constraint, w_ratio, w_critical = params
    
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if machine_options:
            durations = list(machine_options.values())
            min_dur = min(durations)
            avg_dur = sum(durations) / len(durations)
            max_dur = max(durations)
            num_machines = len(durations)
            # Use soft-min for smoother surrogate
            sm_dur = _soft_min(durations, beta=2.0)
        else:
            min_dur = 0.0
            avg_dur = 0.0
            max_dur = 0.0
            num_machines = 1
            sm_dur = 0.0
        
        # Normalized features
        f_mrw = rem_work
        f_mor = float(rem_ops) if rem_ops > 0 else 0.0
        f_min_dur = 1.0 / (sm_dur + 1.0)  # Use soft-min instead of hard min
        f_constraint = 1.0 / (num_machines + 1.0)
        f_ratio = rem_work / (avg_dur + 1.0)
        # Critical path estimate: remaining work / (max duration) as urgency
        f_critical = rem_work / (max_dur + 1.0)
        
        scores[i] = (
            w_mrw * f_mrw +
            w_mor * f_mor +
            w_min_dur * f_min_dur * 10 +
            w_constraint * f_constraint * 5 +
            w_ratio * f_ratio +
            w_critical * f_critical
        )
    
    return int(np.argmax(scores))

def _evaluate_params(ready_operations, time, params):
    """
    Evaluate a set of parameters by simulating a short-horizon greedy dispatch.
    Uses the Hungarian-based dispatch as a lookahead to estimate makespan impact.
    Returns estimated makespan (lower is better).
    """
    if _deadline_exceeded():
        return float('inf')
    
    n = len(ready_operations)
    if n == 0:
        return 0.0
    
    # Quick simulation: apply the param-based dispatch, then estimate remaining makespan
    # Use the soft-min enhanced scoring to estimate quality
    
    w_mrw, w_mor, w_min_dur, w_constraint, w_ratio, w_critical = params
    
    # Score each candidate with the given params
    scores = []
    for i in range(n):
        op = ready_operations[i]
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if machine_options:
            durations = list(machine_options.values())
            sm_dur = _soft_min(durations, beta=2.0)
            avg_dur = sum(durations) / len(durations)
            max_dur = max(durations)
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
        scores.append(score)
    
    # Pick best op
    best_idx = int(np.argmax(scores))
    chosen_op = ready_operations[best_idx]
    
    if chosen_op["machine_options"]:
        sm_dur = _soft_min(list(chosen_op["machine_options"].values()), beta=2.0)
    else:
        sm_dur = 0.0
    
    # Estimate makespan: time + soft_min_duration + estimated remaining
    total_rem_work = sum(op["remaining_work"] for op in ready_operations)
    n_machines = len(set(m for op in ready_operations for m in op["machine_options"]))
    
    # Lower bound on remaining makespan
    base_lb = time + sm_dur + (total_rem_work - sm_dur) / max(1, n_machines)
    
    # Quality adjustment
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    score_range = max_score - min_score if max_score > min_score else 1.0
    quality = (scores[best_idx] - min_score) / score_range if score_range > 0 else 1.0
    
    makespan_est = base_lb * (1.0 + 0.2 * (1.0 - quality))
    
    return makespan_est

def _optimize_params_for_instance(ready_operations, time, n_jobs=10, n_machines=5):
    """
    Optimize dispatch parameters for a specific instance.
    Uses random search over parameter space with Hungarian-informed evaluation.
    Returns best params and best makespan.
    """
    if _deadline_exceeded():
        return None, None
    
    np.random.seed(42)
    
    best_params = None
    best_score = float('inf')
    
    max_trials = 40
    if _time_remaining() < 30:
        max_trials = 10
    
    default_params = (1.5, 0.5, 0.3, 0.5, 0.8, 0.5)
    
    default_score = _evaluate_params(ready_operations, time, default_params)
    if default_score < best_score:
        best_score = default_score
        best_params = default_params
    
    for _ in range(max_trials):
        if _deadline_exceeded():
            break
        
        params = tuple(np.random.uniform(0, 3, size=6))
        score = _evaluate_params(ready_operations, time, params)
        if score < best_score:
            best_score = score
            best_params = params
    
    # Local refinement
    if best_params is not None and _time_remaining() > 20:
        for _ in range(15):
            if _deadline_exceeded():
                break
            noise = np.random.normal(0, 0.25, size=6)
            params = tuple(max(0.0, p + n) for p, n in zip(best_params, noise))
            score = _evaluate_params(ready_operations, time, params)
            if score < best_score:
                best_score = score
                best_params = params
    
    return best_params, best_score

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses Hungarian algorithm for machine assignment combined with param-based scoring.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Check deadline - if too close, fall back to fast simple rule
    if _deadline_exceeded():
        rem_works = [op["remaining_work"] for op in ready_operations]
        return int(np.argmax(rem_works))
    
    # Try Hungarian-based dispatch first (optimization-based)
    # This explicitly accounts for machine contention and load balancing
    try:
        result = _hungarian_dispatch(ready_operations, time)
        # Validate: result should be a valid index
        if 0 <= result < n:
            return result
    except Exception:
        pass
    
    # Fallback to param-based dispatch
    key = _instance_key(ready_operations, time)
    
    if key in _param_cache:
        params = _param_cache[key]
    else:
        if _time_remaining() > 40:
            params, score = _optimize_params_for_instance(ready_operations, time)
            if params is not None:
                _param_cache[key] = params
                global _best_params_global
                if _best_params_global is None:
                    _best_params_global = params
            else:
                params = _best_params_global if _best_params_global else (1.5, 0.5, 0.3, 0.5, 0.8, 0.5)
        else:
            params = _best_params_global if _best_params_global else (1.5, 0.5, 0.3, 0.5, 0.8, 0.5)
    
    return _simple_dispatch(ready_operations, time, params)
# EVOLVE-BLOCK-END
