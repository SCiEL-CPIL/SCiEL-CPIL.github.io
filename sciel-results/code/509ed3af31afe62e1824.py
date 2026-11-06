"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level state to track machine loads across dispatch calls within an instance.
# We reset this when we detect a new simulation instance (time resets or ready_ops structure changes).
_sim_instance_id = None
_machine_load = {}


def _reset_state_if_new_instance(ready_operations, time):
    """Detect a new simulation instance and reset machine load tracking."""
    global _sim_instance_id, _machine_load
    # Heuristic: if time is 0 or very small, and we've seen higher times before, it's a new instance.
    # Also, if the set of jobs changes significantly, it might be a new instance.
    # For simplicity, we'll track the initial set of jobs.
    if time <= 0:
        _sim_instance_id = None
        _machine_load = {}
    
    current_jobs = frozenset(op["job"] for op in ready_operations)
    if _sim_instance_id is None:
        _sim_instance_id = current_jobs
        _machine_load = {}
    else:
        # If jobs have changed drastically, it might be a new instance
        # But in a single instance, the set of jobs is constant.
        # We'll just keep tracking loads.
        pass


def _compute_critical_path(ready_operations):
    """
    Estimate the remaining critical path length for each ready operation.
    We approximate this by remaining_work / remaining_ops (average remaining op duration),
    but we also consider the urgency based on the number of remaining ops.
    """
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        if remaining_ops > 0:
            # Average remaining work per op
            avg_remaining = remaining_work / remaining_ops
        else:
            avg_remaining = 0
        # Critical path estimate: longer remaining path means higher urgency
        # We use a combination of remaining_work and remaining_ops
        # Jobs with more remaining ops and more remaining work are more urgent
        critical_path_estimate = remaining_work + 0.5 * remaining_ops * 10  # 10 is a heuristic scale
        ready_operations[i]["_critical_path"] = critical_path_estimate


def _estimate_machine_load(machine, ready_operations, time):
    """
    Estimate the current load on a machine based on our tracking.
    If we don't have data, we assume a neutral load.
    """
    if machine in _machine_load:
        return _machine_load[machine]
    # Default: assume no load
    return 0.0


def dispatch(ready_operations, time):
    global _machine_load, _sim_instance_id
    
    # Deadline guard: if we're taking too long, fall back to a simple rule
    start_time = _time.time()
    deadline = start_time + 0.5  # 0.5s max for dispatch
    
    if not ready_operations:
        return 0
    
    # Check for new instance
    _reset_state_if_new_instance(ready_operations, time)
    
    n = len(ready_operations)
    
    # Simple fallback if n is very small
    if n == 1:
        return 0
    
    # Compute urgency scores
    # We want to prioritize:
    # 1. Jobs with longer remaining critical paths (to avoid bottlenecks)
    # 2. Operations that can be done on less-loaded machines (to balance load)
    # 3. Shorter processing times on the chosen machine (to reduce completion time)
    
    best_index = 0
    best_score = float('-inf')
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        job = op["job"]
        machine_options = op["machine_options"]
        
        # Get remaining work and ops
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Critical path urgency: jobs with more remaining work and ops are more urgent
        # We use a weighted combination
        if remaining_ops > 0:
            urgency = remaining_work + 0.5 * remaining_ops * 10
        else:
            urgency = remaining_work
        
        # Find the best machine for this operation
        # We want to minimize: machine_load + processing_time
        # But also consider that we should pick the machine that minimizes the completion time
        best_machine_for_op = None
        best_machine_score = float('inf')
        
        for machine, duration in machine_options.items():
            machine_load = _estimate_machine_load(machine, ready_operations, time)
            # Completion time if we assign this op to this machine
            completion_time = machine_load + duration
            # We want to minimize completion time, but also balance load
            # Use a smooth surrogate: min over machines of (machine_load + duration)
            if completion_time < best_machine_score:
                best_machine_score = completion_time
                best_machine_for_op = machine
        
        if best_machine_for_op is None:
            continue
        
        # Final score: higher urgency is better, lower completion time is better
        # Combine them: score = urgency - alpha * completion_time
        # alpha controls the tradeoff
        alpha = 1.0
        score = urgency - alpha * best_machine_score
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_for_op
    
    # Update machine load for the chosen machine
    if best_machine is not None:
        chosen_op = ready_operations[best_index]
        duration = chosen_op["machine_options"].get(best_machine, 0)
        current_load = _machine_load.get(best_machine, 0)
        _machine_load[best_machine] = current_load + duration + time
    
    # Deadline check
    if _time.time() > deadline:
        # Fallback to simple rule
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    return best_index
# EVOLVE-BLOCK-END
