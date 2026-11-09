"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.optimize import linear_sum_assignment

# Global state for machine load tracking
_machine_loads = {}
_last_time = -1.0
_last_job_set = None
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
    return _time.time() - _start_time

def _check_deadline():
    return _get_elapsed() > 145.0  # Safety margin before 150s limit

def _reset_state_if_needed(ready_operations, time):
    global _machine_loads, _last_time, _last_job_set, _start_time
    
    current_jobs = set(op['job'] for op in ready_operations)
    
    if time <= 0.0:
        _machine_loads = {}
        _last_time = time
        _last_job_set = current_jobs
        _start_time = None
        return

    if _last_job_set is not None and len(current_jobs & _last_job_set) == 0 and len(current_jobs) > 0:
        _machine_loads = {}
        _last_time = time
        _last_job_set = current_jobs
        _start_time = None
        return

    _last_time = time
    _last_job_set = current_jobs

def _soft_min(values, beta=2.0):
    """Compute soft-min using log-sum-exp for a smooth surrogate of min."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    v = np.array(values, dtype=np.float64)
    v_min = np.min(v)
    # soft-min = v_min - (1/beta) * log(sum(exp(beta*(v_min - v))))
    # For numerical stability:
    exp_terms = np.exp(beta * (v_min - v))
    log_sum = np.log(np.sum(exp_terms))
    return v_min - log_sum / beta

def dispatch(ready_operations, time):
    global _machine_loads, _last_time, _last_job_set
    
    if not ready_operations:
        return 0
    
    _reset_state_if_needed(ready_operations, time)
    
    # Deadline guard: if we're running out of time, fall back to simple heuristic
    if _check_deadline():
        best_idx = 0
        best_machine = None
        best_dur = float('inf')
        for i, op in enumerate(ready_operations):
            for m, dur in op['machine_options'].items():
                if dur < best_dur:
                    best_dur = dur
                    best_idx = i
                    best_machine = m
        if best_machine is not None:
            _machine_loads[best_machine] = _machine_loads.get(best_machine, 0.0) + best_dur
            return (best_idx, best_machine)
        return 0
    
    n_ops = len(ready_operations)
    
    # Get all unique machines involved in these ready operations
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op['machine_options'].keys())
    
    if not all_machines:
        return 0

    # Initialize loads for any new machines
    for m in all_machines:
        if m not in _machine_loads:
            _machine_loads[m] = 0.0

    best_idx = -1
    best_machine = None
    best_score = -float('inf')
    
    # Soft-min temperature: lower beta = smoother (more uniform), higher beta = closer to hard min
    beta = 3.0
    
    for i, op in enumerate(ready_operations):
        rem_work = op['remaining_work']
        rem_ops = op['remaining_ops']
        
        # Collect resulting loads for all machine options
        resulting_loads = []
        durations = []
        for m, dur in op['machine_options'].items():
            current_load = _machine_loads.get(m, 0.0)
            resulting_loads.append(current_load + dur)
            durations.append(dur)
        
        if not resulting_loads:
            continue
        
        # Soft-min of resulting loads: smooth surrogate that considers all options
        soft_min_load = _soft_min(resulting_loads, beta=beta)
        
        # Hard min for tie-breaking / argmin
        min_load = min(resulting_loads)
        min_idx = resulting_loads.index(min_load)
        best_m_for_op = list(op['machine_options'].keys())[min_idx]
        
        # Min duration across machines
        min_duration = min(durations)
        
        # Flexibility: number of machine options
        flexibility = len(op['machine_options'])
        
        # Composite score:
        # - High remaining work = urgent (finish big jobs early)
        # - Low soft-min load = load balancing (smooth, considers alternatives)
        # - Small min duration = efficiency
        # - Flexibility bonus = prefer more flexible ops (less constraint risk)
        
        # Normalize rem_work to avoid scale issues
        # Use a weighted combination tuned for makespan minimization
        w_work = 1.0
        w_load = 0.3
        w_dur = 0.05
        w_flex = 0.02
        
        score = w_work * rem_work - w_load * soft_min_load - w_dur * min_duration + w_flex * flexibility
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op
            
    if best_idx == -1:
        return 0
        
    # Update the global state
    selected_op = ready_operations[best_idx]
    dur = selected_op['machine_options'][best_machine]
    _machine_loads[best_machine] += dur
    
    # Return (index, machine) to explicitly control assignment
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
