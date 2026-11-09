"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_start_time = _time.monotonic()
_DEADLINE = 115.0
_SAFETY_MARGIN = 5.0


def dispatch(ready_operations, time):
    # Check deadline guard
    if _time.monotonic() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick the operation with maximum remaining work
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Extract features
    rw_list = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=float)
    rem_ops_list = np.array([op.get("remaining_ops", 1) for op in ready_operations], dtype=float)
    
    # Minimum duration among machine options for each operation
    min_dur_list = np.array([min(op["machine_options"].values()) if op["machine_options"] else 0.0 for op in ready_operations], dtype=float)
    
    # Soft-min over machine options for a smooth estimate
    def soft_min(x, tau=0.1):
        if len(x) == 0:
            return 0.0
        neg_x = -x / tau
        max_neg_x = np.max(neg_x)
        if max_neg_x > 700:
            return float(np.min(x))
        log_sum = np.log(np.sum(np.exp(neg_x - max_neg_x))) + max_neg_x
        return -log_sum * tau
    
    soft_min_durs = np.array([soft_min(np.array(list(op["machine_options"].values()), dtype=float)) for op in ready_operations])
    
    # Number of machine options (flexibility)
    n_opts_list = np.array([len(op["machine_options"]) for op in ready_operations], dtype=float)
    
    # Critical path proxy: average remaining operation duration
    avg_dur = rw_list / np.maximum(rem_ops_list, 1.0)
    
    # SPT term: reciprocal of soft-min duration (favors short operations)
    eps = 1e-6
    spt_term = 1.0 / (soft_min_durs + eps)
    
    # LPT term: favors long operations (bottleneck elimination)
    # We use a moderate weight for LPT to balance with SPT
    
    # Normalize all components to [0, 1]
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-9:
            return np.full_like(arr, 0.5)
        return (arr - mn) / rng
    
    rw_norm = normalize(rw_list)
    dur_norm = normalize(soft_min_durs)
    ops_norm = normalize(rem_ops_list)
    spt_norm = normalize(spt_term)
    avg_norm = normalize(avg_dur)
    flex_norm = normalize(n_opts_list)
    
    # Composite score:
    # - SPT (local efficiency): strong weight to reduce idle time
    # - Remaining work (global urgency): moderate weight for critical path
    # - LPT (bottleneck awareness): moderate weight to schedule long ops early
    # - Average duration (critical path proxy): secondary
    # - Flexibility (more options): small weight
    scores = (
        1.2 * spt_norm +
        0.8 * rw_norm +
        0.4 * (1.0 - dur_norm) +  # LPT: higher duration gets higher score
        0.3 * avg_norm +
        0.1 * flex_norm
    )
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
