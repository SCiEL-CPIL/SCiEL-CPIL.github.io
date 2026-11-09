import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0

# Stateful tracking of machine loads
# Key: machine_id, Value: total time scheduled on that machine so far
_machine_loads = {}

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    best_idx = 0
    best_completion = float('inf')
    best_machine = None

    n = len(ready_operations)

    for i, op in enumerate(ready_operations):
        options = op['machine_options']
        if not options:
            continue

        # For each machine option, calculate the projected completion time
        # projected_completion = current_load_of_machine + duration
        # We want to minimize this to balance load and reduce makespan
        
        for machine, duration in options.items():
            current_load = _machine_loads.get(machine, 0)
            projected_completion = current_load + duration
            
            if projected_completion < best_completion:
                best_completion = projected_completion
                best_idx = i
                best_machine = machine

    # Update the stateful load for the chosen machine
    # Note: We must be careful here. The 'time' parameter is the current simulation time.
    # However, the dispatch rule is called when a machine becomes free.
    # If we simply add duration to the load, we are tracking total busy time.
    # This is a valid proxy for makespan minimization in many JSS heuristics.
    
    if best_machine is not None:
        # We need the duration used to update the load correctly
        # Find the duration for best_machine in the chosen op
        chosen_op = ready_operations[best_idx]
        chosen_duration = chosen_op['machine_options'].get(best_machine, 0)
        _machine_loads[best_machine] = _machine_loads.get(best_machine, 0) + chosen_duration

    return (best_idx, best_machine)
