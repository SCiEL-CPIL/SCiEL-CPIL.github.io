"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = 120.0  # Seconds

def dispatch(ready_operations, time):
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - 1.0:
        return 0

    if not ready_operations:
        return 0

    best_score = -1.0
    best_index = 0
    
    for i, op in enumerate(ready_operations):
        # Calculate minimum duration among all machine options
        # This acts as a proxy for "efficiency" or "ease of scheduling"
        min_dur = min(op["machine_options"].values())
        
        # Remaining work represents urgency
        rem_work = op["remaining_work"]
        
        # Heuristic: Urgency adjusted by efficiency.
        # High remaining work is good, but we penalize if the operation is slow (high min_dur).
        # Using division to balance the scales.
        score = rem_work / (min_dur + 1e-9)
        
        if score > best_score:
            best_score = score
            best_index = i

    return best_index
# EVOLVE-BLOCK-END
