"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: keep a safety margin before the 120s wall-clock limit
_DEADLINE = 120.0
_START_TIME = _time.time()
_SAFETY_MARGIN = 2.0  # seconds


def _elapsed():
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    Returns index of the ready operation to start.
    Optionally returns (index, machine) for machine selection.
    
    Strategy: Composite scoring combining:
    1. Remaining work (more work = higher priority)
    2. Machine flexibility (more options = higher priority, reduces bottlenecks)
    3. Shortest processing time on best machine (SPT-like)
    4. Remaining ops in job (jobs closer to completion get priority to free resources)
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Check deadline
    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: return the operation with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    scores = np.zeros(n)
    
    # Compute components
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # Flexibility: number of machine options
    flexibilities = np.array([len(op["machine_options"]) for op in ready_operations], dtype=float)
    
    # Min duration across machines (best-case processing time)
    min_durations = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=float)
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    nw_norm = normalize(remaining_works)
    ro_norm = normalize(remaining_ops)
    fl_norm = normalize(flexibilities)
    # For min duration, we want shorter = higher priority, so invert
    md_norm = 1.0 - normalize(min_durations)
    
    # Weighted composite score
    # Weights tuned to balance competing objectives
    w_ro = 0.35  # remaining work
    w_fl = 0.25  # flexibility
    w_md = 0.25  # min duration (shorter is better)
    w_ro_ops = 0.15  # remaining ops (fewer = closer to done = higher priority)
    
    # For remaining ops, fewer ops means closer to completion, so we want to prioritize
    # jobs that are closer to finishing to free up resources
    # Actually, let's think about this differently:
    # - High remaining work: prioritize (long jobs)
    # - High flexibility: prioritize (less likely to block)
    # - Low min duration: prioritize (quick wins)
    # - Low remaining ops: prioritize (finish jobs to free resources)
    
    scores = (w_ro * nw_norm + 
              w_fl * fl_norm + 
              w_md * md_norm + 
              w_ro_ops * (1.0 - ro_norm))
    
    best_idx = int(np.argmax(scores))
    
    # Optionally select the best machine (shortest duration)
    best_op = ready_operations[best_idx]
    best_machine = min(best_op["machine_options"], key=best_op["machine_options"].get)
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
