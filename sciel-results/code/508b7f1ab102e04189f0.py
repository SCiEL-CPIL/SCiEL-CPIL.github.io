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
    1. Primary: Job with most remaining work (LRPT/MWR).
    2. Secondary: Operation with fewer machine options (constrained first).
    3. Tertiary: Minimize effective duration (min duration).
    
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

    # Pre-calculate features for each ready operation
    scores = [0.0] * n
    best_machines = [None] * n
    best_durations = [0.0] * n

    # Weights for the composite heuristic
    W_RW = 10.0       # Weight for Remaining Work (Job urgency)
    W_FLEX = 2.0      # Weight for Flexibility (1/num_options)
    W_DUR = 0.5       # Weight for Duration (penalty for long min duration)

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        durations = op["machine_options"]
        
        if not durations:
            scores[i] = -1e9
            best_machines[i] = None
            best_durations[i] = 0.0
            continue

        best_m, best_d = _best_machine_for_op(op)
        best_machines[i] = best_m
        best_durations[i] = best_d
        
        num_opts = len(durations)
        flex_score = 1.0 / num_opts if num_opts > 0 else 0.0
        
        # Composite Score:
        # High remaining work is good.
        # High flexibility (few options) is good (schedule constrained ops early).
        # Low min duration is good (clears quickly, less blocking).
        score = W_RW * rw + W_FLEX * flex_score - W_DUR * best_d
        scores[i] = score

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
