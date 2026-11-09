"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Stateful tracking of machine loads
_machine_loads = {}

def dispatch(ready_operations, time):
    """
    Stateful machine-centric dispatch rule.
    Strategy: Minimize projected completion time by favoring operations that can finish
    earliest on the least-loaded eligible machine. Uses a small tie-breaker for remaining work.
    """
    global _machine_loads
    
    if not ready_operations:
        return 0

    # Ensure machine loads are initialized
    for op in ready_operations:
        for machine in op["machine_options"]:
            if machine not in _machine_loads:
                _machine_loads[machine] = 0.0

    best_idx = 0
    best_score = float('inf')
    best_min_duration = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Find the machine that gives the earliest completion for this operation
        min_completion = float('inf')
        min_duration = float('inf')
        
        for machine, duration in op["machine_options"].items():
            load = _machine_loads.get(machine, 0.0)
            completion = load + duration
            if completion < min_completion:
                min_completion = completion
                min_duration = duration

        # Score: primarily earliest completion, secondarily shortest duration (SPT-like tie-break)
        # We want to minimize this score.
        score = min_completion * 1.0 + min_duration * 0.1
        
        if score < best_score:
            best_score = score
            best_idx = i
            best_min_duration = min_duration

    # Update the load of the selected machine for the next dispatch call
    # We assume the operation starts at 'time' but the load tracking is relative to dispatch order
    # To keep it simple and consistent with the "load" concept, we add the duration to the chosen machine.
    # Note: This is an approximation. A more precise model would track absolute time, but
    # relative load balancing is often effective for makespan reduction in JSS.
    selected_op = ready_operations[best_idx]
    chosen_machine = min(selected_op["machine_options"], key=lambda m: _machine_loads.get(m, 0.0) + selected_op["machine_options"][m])
    
    # Update the load
    _machine_loads[chosen_machine] += selected_op["machine_options"][chosen_machine]

    return best_idx

# EVOLVE-BLOCK-END
