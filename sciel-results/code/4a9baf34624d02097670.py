"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def dispatch(ready_operations, time):
    """Pick the ready operation to start next, optionally with machine choice.
    
    Strategy: Workload-weighted tightness — prioritize operations with high
    remaining work that also have short processing times on their best machine.
    Returns (index, machine) tuple to also select the machine.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick shortest-duration machine
        best_m = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        return (0, best_m)
    
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback: just pick max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute scores: combine remaining work with tightness (shortest proc time)
    # For each ready op, find best (shortest) machine duration
    scores = []
    best_machines = []
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        min_dur = min(mo.values())
        # Tightness: high remaining work / short processing time = urgent
        # Normalize to avoid division by zero
        tightness = op["remaining_work"] / (min_dur + 1e-9)
        scores.append(tightness)
        best_machines.append(min(mo, key=mo.get))
    
    scores_arr = np.array(scores, dtype=np.float64)
    best_idx = int(np.argmax(scores_arr))
    return (best_idx, best_machines[best_idx])
# EVOLVE-BLOCK-END
