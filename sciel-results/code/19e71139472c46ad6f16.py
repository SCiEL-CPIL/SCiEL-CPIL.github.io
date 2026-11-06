"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Global deadline guard
_START_TIME = time_module.time()
_DEADLINE = _START_TIME + 110.0  # 110s limit to be safe within 120s

def dispatch(ready_operations, time):
    # Check deadline
    if time_module.time() > _DEADLINE:
        return 0
    
    # Machine-contention avoidance heuristic:
    # Score each operation by the ratio of min duration to sum of durations.
    # This prioritizes operations that can be executed on less loaded machines,
    # reducing idle time and contention.
    scores = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        
        if not machine_options:
            # Fallback: if no machine options, give a neutral score
            score = 0.0
        else:
            durations = list(machine_options.values())
            min_duration = min(durations)
            sum_durations = sum(durations)
            
            if sum_durations <= 0:
                score = 1.0
            else:
                # Ratio of min to sum: higher when operation has fewer/shorter alternatives
                # This favors operations with less machine contention
                score = min_duration / sum_durations
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
