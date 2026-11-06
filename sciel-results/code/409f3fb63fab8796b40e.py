"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time_mod

# Deadline guard
_DEADLINE = None
_TIME_LIMIT = 145.0  # safety margin before 150s kill

def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = _time_mod.monotonic() + _TIME_LIMIT
    return _DEADLINE

def _time_exceeded():
    return _time_mod.monotonic() > _DEADLINE

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    Returns (index, machine) where index is the ready operation to start
    and machine is the chosen machine from its options.
    """
    if not ready_operations:
        return 0, 0

    _ensure_deadline()

    # If time is running out, fall back to a fast heuristic
    if _time_exceeded():
        # Fast fallback: pick op with most remaining work, shortest machine
        best_idx = 0
        best_rw = ready_operations[0]["remaining_work"]
        for i, op in enumerate(ready_operations):
            if op["remaining_work"] > best_rw:
                best_rw = op["remaining_work"]
                best_idx = i
        mo = ready_operations[best_idx]["machine_options"]
        best_m = min(mo.keys(), key=lambda m: mo[m])
        return best_idx, best_m

    n = len(ready_operations)
    
    # Extract data for vectorized computation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    # Compute min and max durations for each operation
    min_durs = np.zeros(n)
    max_durs = np.zeros(n)
    for i, op in enumerate(ready_operations):
        durs = np.array(list(op["machine_options"].values()))
        min_durs[i] = np.min(durs)
        max_durs[i] = np.max(durs)
    
    # Multi-criteria scoring:
    # 1. Remaining work (higher is better) - normalized
    # 2. Remaining ops (higher is better) - indicates more work ahead
    # 3. Min duration (lower is better) - quick completion
    # 4. Range (max-min) (lower is better) - less variability
    
    # Normalize each criterion to [0,1]
    rw_min, rw_max = np.min(remaining_works), np.max(remaining_works)
    ro_min, ro_max = np.min(remaining_ops), np.max(remaining_ops)
    md_min, md_max = np.min(min_durs), np.max(min_durs)
    
    # Avoid division by zero
    rw_norm = (remaining_works - rw_min) / (rw_max - rw_min + 1e-9)
    ro_norm = (remaining_ops - ro_min) / (ro_max - ro_min + 1e-9)
    md_norm = (min_durs - md_min) / (md_max - md_min + 1e-9)  # lower is better, so invert
    
    # Weighted score: emphasize remaining work, then remaining ops, then short min duration
    # Higher score = better
    weights = np.array([0.5, 0.2, 0.3])
    scores = weights[0] * rw_norm + weights[1] * ro_norm + weights[2] * (1.0 - md_norm)
    
    best_idx = int(np.argmax(scores))
    
    # Select machine: prefer the one with shortest duration, but also consider
    # load balancing - if we can estimate machine loads, prefer less loaded
    # For simplicity, use shortest processing time with a small tiebreaker
    best_op = ready_operations[best_idx]
    machine_options = best_op["machine_options"]
    
    # Find machine with minimum duration
    best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
    
    return best_idx, best_machine
# EVOLVE-BLOCK-END
