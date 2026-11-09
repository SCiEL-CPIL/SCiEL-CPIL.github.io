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
    
    # Use soft-min for smooth surrogate of minimum duration
    # and soft-max for smooth surrogate of maximum remaining work
    
    n = len(ready_operations)
    rw_list = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=float)
    min_dur_list = np.array([min(op["machine_options"].values()) if op["machine_options"] else 0 for op in ready_operations], dtype=float)
    rem_ops_list = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=float)
    
    # Soft-min for processing times (lower is better, so we want to minimize)
    # Use log-sum-exp trick for numerical stability
    # soft_min(x) = -logsumexp(-x)
    def soft_min(x, tau=1.0):
        if len(x) == 0:
            return 0.0
        neg_x = -x / tau
        max_neg_x = np.max(neg_x)
        if max_neg_x > 700:  # numerical safety
            return float(np.min(x))
        log_sum = np.log(np.sum(np.exp(neg_x - max_neg_x))) + max_neg_x
        return -log_sum * tau
    
    # Soft-max for remaining work (higher is better)
    def soft_max(x, tau=1.0):
        if len(x) == 0:
            return 0.0
        x_scaled = x / tau
        max_x = np.max(x_scaled)
        if max_x > 700:  # numerical safety
            return float(np.max(x))
        log_sum = np.log(np.sum(np.exp(x_scaled - max_x))) + max_x
        return log_sum * tau
    
    # Compute soft-min for each operation's machine options
    # This gives a smooth estimate of the minimum processing time
    soft_min_durs = np.array([soft_min(np.array(list(op["machine_options"].values()), dtype=float), tau=0.5) for op in ready_operations])
    
    # Combine scores with smooth surrogates
    # Weight: remaining work (max), minus soft-min duration, plus urgency
    rw_max = soft_max(rw_list, tau=1.0)
    rem_ops_max = soft_max(rem_ops_list, tau=1.0)
    
    # Normalize components
    rw_range = np.max(rw_list) - np.min(rw_list) if np.max(rw_list) > np.min(rw_list) else 1.0
    dur_range = np.max(soft_min_durs) - np.min(soft_min_durs) if np.max(soft_min_durs) > np.min(soft_min_durs) else 1.0
    ops_range = np.max(rem_ops_list) - np.min(rem_ops_list) if np.max(rem_ops_list) > np.min(rem_ops_list) else 1.0
    
    rw_norm = (rw_list - np.min(rw_list)) / rw_range
    dur_norm = (soft_min_durs - np.min(soft_min_durs)) / dur_range
    ops_norm = (rem_ops_list - np.min(rem_ops_list)) / ops_range
    
    # Score: prioritize high remaining work, low processing time, high urgency
    scores = rw_norm * 1.0 - dur_norm * 0.6 + ops_norm * 0.2
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
