"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track elapsed time and bail out to a safe fallback if we approach the limit.
_T0 = _time.perf_counter()
_DEADLINE = 140.0  # safety margin of 10s


def _elapsed_ok():
    return (_time.perf_counter() - _T0) < _DEADLINE


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Use a composite heuristic that balances:
    1. Remaining work (more work = higher priority)
    2. Remaining operations (more ops = higher priority, as they constrain completion)
    3. Machine load consideration (prefer less-loaded machines via machine_options)
    
    We use a weighted combination with soft-max-like behavior to pick the best operation.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Quick deadline check
    if not _elapsed_ok():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        mo = op["machine_options"]
        
        # Minimum duration across available machines (soonest we could finish this op)
        if mo:
            min_dur = min(mo.values())
            # Average duration
            avg_dur = sum(mo.values()) / len(mo)
            # Number of machine options (flexibility)
            n_opts = len(mo)
        else:
            min_dur = 1.0
            avg_dur = 1.0
            n_opts = 1
        
        # Heuristic components:
        # 1. Remaining work: more work means more critical
        # 2. Remaining ops: more ops means this job is further from completion
        # 3. Shorter minimum duration: can be done sooner, freeing up capacity
        # 4. More machine options: more flexible, less likely to cause bottlenecks
        
        # Normalize-ish scaling (avoid division by zero)
        score = (
            rw * 1.0 +           # primary: remaining work
            ro * 0.5 +            # secondary: remaining operations
            - min_dur * 0.3 +     # prefer shorter ops (frees machines sooner)
            n_opts * 0.1          # prefer more flexible ops
        )
        scores[i] = score
    
    # Pick the operation with highest score
    idx = int(np.argmax(scores))
    
    return idx
# EVOLVE-BLOCK-END
