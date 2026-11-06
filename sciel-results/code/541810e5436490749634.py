"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track machine loads across dispatch calls for this instance
_machine_loads = {}
_instance_id = 0

def _get_machine_load(machine):
    """Get current estimated load for a machine."""
    return _machine_loads.get(machine, 0)

def _update_machine_load(machine, duration):
    """Update machine load after assignment."""
    _machine_loads[machine] = _machine_loads.get(machine, 0) + duration

def dispatch(ready_operations, time):
    global _machine_loads, _instance_id
    
    # If no ready operations, return -1 or handle edge case
    if not ready_operations:
        return -1
    
    # Reset machine loads if we detect a new instance (heuristic: if time is 0 or very small)
    # This is a simple heuristic; in practice, the evaluator likely calls dispatch
    # for each event, and time increases. We'll assume loads persist within an instance.
    # For safety, if time < 1e-6, we might be at the start of a new instance.
    if time < 1e-6:
        _machine_loads.clear()
        _instance_id += 1
    
    best_score = -np.inf
    best_index = 0
    best_machine = None
    
    for idx, op in enumerate(ready_operations):
        # Estimate critical path contribution for this operation
        # Sum of minimum durations for all remaining operations in this job
        machine_options = op["machine_options"]
        if not machine_options:
            continue
        
        # Find the best machine for this operation considering load
        best_op_score = -np.inf
        best_mach_for_op = None
        
        for machine, duration in machine_options.items():
            current_load = _get_machine_load(machine)
            # The completion time of this operation if assigned to this machine
            # is approximately current_load + duration
            # We want to minimize this, but also consider the critical path
            completion_time = current_load + duration
            
            # Critical path estimate: this operation's completion + remaining work after it
            # remaining_work includes this op's duration, so subtract it
            remaining_after = op["remaining_work"] - duration
            
            # Total critical path contribution
            cp_contribution = completion_time + remaining_after
            
            # We want to minimize cp_contribution, so score is negative
            score = -cp_contribution
            
            if score > best_op_score:
                best_op_score = score
                best_mach_for_op = machine
        
        # If we found a valid machine, use its score
        if best_mach_for_op is not None:
            # Add a small bonus for operations with more remaining ops (urgency)
            urgency_bonus = 0.1 * op["remaining_ops"]
            final_score = best_op_score + urgency_bonus
            
            if final_score > best_score:
                best_score = final_score
                best_index = idx
                best_machine = best_mach_for_op
    
    # Update the machine load for the selected operation
    if best_machine is not None:
        selected_op = ready_operations[best_index]
        duration = selected_op["machine_options"][best_machine]
        _update_machine_load(best_machine, duration)
    
    return best_index
# EVOLVE-BLOCK-END
