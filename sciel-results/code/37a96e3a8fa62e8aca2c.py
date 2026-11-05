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
    1. Primary: Job with most remaining work (LRPT) to prioritize critical path.
    2. Secondary: Job with highest average remaining processing time (ARPT) to prioritize jobs with fewer, larger operations.
    3. Tertiary: Operation with shortest minimum duration (SPT) to release resources quickly.
    4. Machine: Machine with shortest duration for the chosen operation (SPT).
    
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

    scores = []
    best_machines = []
    
    for op in ready_operations:
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        durations = op["machine_options"]
        
        # Handle remaining_ops type robustly
        if isinstance(ro, (int, float)):
            num_ops = ro
        else:
            try:
                num_ops = len(ro)
            except TypeError:
                num_ops = 1
        
        if not durations:
            scores.append(-1)
            best_machines.append(None)
            continue
            
        min_dur = min(durations.values())
        
        # Score components
        # 1. Total remaining work (LRPT) - dominant
        # 2. Average remaining work per op (ARPT) - secondary, higher is better (fewer, larger ops)
        # 3. Minimum duration of current op (SPT) - tertiary, lower is better (faster release)
        
        avg_work = rw / max(1, num_ops)
        
        # We want to maximize score.
        # rw * 1e6 dominates.
        # avg_work * 1e4 is secondary.
        # -min_dur is tertiary (SPT).
        score = rw * 1e6 + avg_work * 1e4 - min_dur
        
        scores.append(score)
        
        best_m, _ = _best_machine_for_op(op)
        best_machines.append(best_m)

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
