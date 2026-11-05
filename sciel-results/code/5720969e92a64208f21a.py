"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE_MARGIN = 5  # seconds before the 150s kill; be conservative


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Primary: Job with most remaining work (LRPT).
    2. Secondary: Minimize effective duration, which includes a penalty for machine contention.
       Machine contention is estimated by counting how many ready operations consider this machine
       as their best (shortest duration) option.
    3. Machine: Select the machine that minimizes the effective duration (duration + contention penalty).
    
    Return: (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op)
        return (0, best_m)

    # Deadline guard: if we are too close to the limit, return a quick heuristic
    if _elapsed() > 145:
        # Quick fallback: pick first op, min duration machine
        op = ready_operations[0]
        best_m, _ = _best_machine_for_op(op)
        return (0, best_m)

    # Step 1: Determine the "best" machine (min duration) for each ready operation
    # and count contention on those machines.
    best_machines = [None] * n
    best_durations = [0.0] * n
    machine_contention = {}

    for i, op in enumerate(ready_operations):
        durations = op["machine_options"]
        if not durations:
            best_machines[i] = None
            best_durations[i] = 0.0
            continue
        
        best_m, best_d = _best_machine_for_op(op)
        best_machines[i] = best_m
        best_durations[i] = best_d
        
        # Count how many ops prefer this machine
        if best_m is not None:
            machine_contention[best_m] = machine_contention.get(best_m, 0) + 1

    # Step 2: Calculate scores
    # We want to maximize score.
    # Primary: Remaining Work (LRPT)
    # Secondary: Minimize Effective Duration
    # Effective Duration = Min Duration + Contention Penalty
    # We subtract Effective Duration from Score.
    
    # Weight for LRPT dominance. 
    # If RW is 100 and Dur is 10, we want RW to be more important.
    # Let's try a weight of 10.0 for RW.
    RW_WEIGHT = 10.0
    CONTENTION_WEIGHT = 1.0  # Each contender adds 1.0 to the effective duration
    
    scores = []
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        best_m = best_machines[i]
        best_d = best_durations[i]
        
        if best_m is None:
            scores.append(-1)
            continue
            
        contention = machine_contention.get(best_m, 0)
        # The penalty is proportional to the number of other ops fighting for this machine.
        # We subtract 1 because the op itself is counted in contention, so "other" ops is contention - 1.
        # Or just use contention directly as a proxy for load.
        penalty = contention * CONTENTION_WEIGHT
        effective_dur = best_d + penalty
        
        score = rw * RW_WEIGHT - effective_dur
        scores.append(score)

    # Find index with max score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    if best_machine is None:
        return best_idx
    return (best_idx, best_machine)


def _best_machine_for_op(op):
    """
    For a given operation, select the machine that minimizes its duration.
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    best_machine = min(durations, key=lambda m: durations[m])
    return best_machine, durations[best_machine]
# EVOLVE-BLOCK-END
