"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


def dispatch(ready_operations, time):
    # Check deadline guard
    if _time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick the operation with minimum processing time on any machine
        min_durations = [min(op["machine_options"].values()) for op in ready_operations]
        return int(np.argmin(min_durations))
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Extract features
    rws = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=np.float64)
    min_durs = np.array([min(op["machine_options"].values()) if op["machine_options"] else 0 for op in ready_operations], dtype=np.float64)
    rem_ops = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=np.float64)
    
    # Normalize features to [0, 1] range for consistent weighting
    rw_min, rw_max = rws.min(), rws.max()
    dur_min, dur_max = min_durs.min(), min_durs.max()
    ops_min, ops_max = rem_ops.min(), rem_ops.max()
    
    if rw_max > rw_min:
        rw_norm = (rws - rw_min) / (rw_max - rw_min)
    else:
        rw_norm = np.ones(n)
    
    if dur_max > dur_min:
        dur_norm = (min_durs - dur_min) / (dur_max - dur_min)
    else:
        dur_norm = np.ones(n)
    
    if ops_max > ops_min:
        ops_norm = (rem_ops - ops_min) / (ops_max - ops_min)
    else:
        ops_norm = np.ones(n)
    
    # Weights: prioritize remaining work (critical path), then short processing time,
    # then urgency from remaining operations
    # Use a smooth combination that emphasizes the dominant factor
    w_rw = 0.6
    w_dur = 0.3
    w_ops = 0.1
    
    # Score: high remaining work is good, short duration is good (invert), high remaining ops is good
    scores = w_rw * rw_norm - w_dur * dur_norm + w_ops * ops_norm
    
    # Add a small tie-breaker based on index to ensure determinism
    scores += np.arange(n) * 1e-10
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
