"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 145.0  # safety margin before 150s kill
_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.monotonic()
    return _START_TIME


def _is_deadline_exceeded():
    return (_time.monotonic() - _ensure_start()) > _DEADLINE


def dispatch(ready_operations, time):
    """
    Multi-criteria dispatch rule for job-shop scheduling.
    
    Combines:
    1. WSPT-like ratio: remaining_work / min_machine_duration (urgency)
    2. Critical path: remaining_ops (jobs with more ops left are more critical)
    3. Machine load balancing: prefer machines that are less loaded
    
    Returns index of the ready operation to start.
    """
    _ensure_start()
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Compute features for each ready operation
    scores = np.zeros(n)
    
    # Feature 1: WSPT-like ratio (work per unit time) - higher is more urgent
    wspt_values = np.zeros(n)
    min_durations = np.zeros(n)
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 1.0
        min_durations[i] = min_dur
        wspt_values[i] = op["remaining_work"] / max(min_dur, 1e-9)
    
    # Normalize WSPT to [0, 1]
    wspt_max = np.max(wspt_values)
    wspt_min = np.min(wspt_values)
    if wspt_max > wspt_min:
        wspt_norm = (wspt_values - wspt_min) / (wspt_max - wspt_min)
    else:
        wspt_norm = np.ones(n) / n
    
    # Feature 2: Remaining operations count (critical path length)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    ops_max = np.max(remaining_ops)
    ops_min = np.min(remaining_ops)
    if ops_max > ops_min:
        ops_norm = (remaining_ops - ops_min) / (ops_max - ops_min)
    else:
        ops_norm = np.ones(n) / n
    
    # Feature 3: Machine load balancing - prefer operations on less-loaded machines
    # We estimate load by checking which machines are busy
    # Since we don't have direct machine state, we use a heuristic:
    # Operations with more machine options are more flexible (lower priority for specific machine)
    # Operations with fewer options are more constrained (higher priority)
    flexibility = np.zeros(n)
    for i, op in enumerate(ready_operations):
        n_opts = len(op["machine_options"])
        # Fewer options = more constrained = higher priority
        flexibility[i] = 1.0 / max(n_opts, 1)
    
    flex_max = np.max(flexibility)
    flex_min = np.min(flexibility)
    if flex_max > flex_min:
        flex_norm = (flexibility - flex_min) / (flex_max - flex_min)
    else:
        flex_norm = np.ones(n) / n
    
    # Combined score with weights
    # WSPT is the primary driver, critical path second, flexibility third
    w1 = 0.5  # WSPT
    w2 = 0.35  # Remaining ops
    w3 = 0.15  # Flexibility/constraint
    
    scores = w1 * wspt_norm + w2 * ops_norm + w3 * flex_norm
    
    # Deadline guard: if we're running long, just return best so far
    if _is_deadline_exceeded():
        return int(np.argmax(scores))
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
