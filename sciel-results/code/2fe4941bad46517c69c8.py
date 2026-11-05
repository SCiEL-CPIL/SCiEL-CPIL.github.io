"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard
_START_TIME = _time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _check_deadline():
    return (_time.time() - _START_TIME) < _TIME_LIMIT


def dispatch(ready_operations, time):
    """Pick which ready operation to start next.
    
    Uses a composite priority:
    - Primary: shortest processing time on best machine (SPT)
    - Secondary: fewer remaining operations (closer to completion)
    - Tertiary: more remaining work (criticality)
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute min duration for each operation
    min_durs = []
    for op in ready_operations:
        opts = op["machine_options"]
        min_durs.append(min(opts.values()) if opts else 0)
    
    # SPT: pick operation with shortest min duration
    # Tiebreak: fewer remaining ops, then more remaining work
    best_idx = 0
    best_key = None
    
    for i, op in enumerate(ready_operations):
        dur = min_durs[i]
        rem_ops = op["remaining_ops"]
        rem_work = op["remaining_work"]
        
        # We want to minimize dur, then minimize rem_ops, then maximize rem_work
        # So key = (dur, rem_ops, -rem_work)
        key = (dur, rem_ops, -rem_work)
        
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
