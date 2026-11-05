"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# We'll use a composite heuristic. Since dispatch is called per-event,
# we implement a fast, strong rule.
#
# Heuristic: 
#   score = remaining_work - alpha * soft_min(duration) + beta * remaining_ops
#   Higher score = pick first.
#   If we can return (index, machine), we pick the machine with min duration.
#
# Let's use a weighted combination that emphasizes remaining_work heavily,
# with a small bonus for choosing shorter machines.

_ALPHA = 0.1  # weight for machine duration preference
_BETA = 0.01  # weight for remaining ops urgency
_SOFTMIN_TEMP = 5.0  # temperature for soft-min (lower = sharper)

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        # Return index and best machine
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)
    
    scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        mw = op["remaining_work"]
        remaining_ops = op.get("remaining_ops", 0)
        
        # Get machine durations
        durations = list(op["machine_options"].values())
        
        # Use soft-min for smoother machine preference
        # soft_min = -log(sum(exp(-d/temp)))/temp
        # For numerical stability, subtract min first
        min_dur = min(durations)
        shifted = [d - min_dur for d in durations]
        # Compute soft-min
        exp_terms = [np.exp(-s / _SOFTMIN_TEMP) for s in shifted]
        sum_exp = sum(exp_terms)
        soft_min_val = min_dur - _SOFTMIN_TEMP * np.log(sum_exp)
        
        # Composite score: remaining_work is dominant
        # Shorter machines are better, so subtract alpha * soft_min
        score = float(mw) - _ALPHA * float(soft_min_val)
        
        # Add small bonus for jobs with more remaining ops (keep them flowing)
        score += _BETA * float(remaining_ops)
        
        scores.append(score)
        # Track the best (min duration) machine for this op
        best_machines.append(min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m]))
    
    scores = np.array(scores)
    idx = int(np.argmax(scores))
    
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
