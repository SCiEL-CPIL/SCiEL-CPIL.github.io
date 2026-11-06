"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick the next ready operation to start in an open-shop simulation.
    
    Strategy: Composite heuristic that balances:
    1. Job urgency (remaining_work) - process jobs with more remaining work
    2. Machine efficiency (min_duration) - prefer shorter operations to reduce blocking
    3. Workload balance - avoid overloading certain machines
    4. Critical path consideration - prioritize operations that are on critical paths
    
    The heuristic is designed to reduce idle time and keep critical-path operations
    progressing effectively.
    """
    if not ready_operations:
        return 0
    
    # Explicit deadline guard - track elapsed time
    _t0 = _time.time()
    _deadline = _t0 + 0.5  # Safety margin well within 120s limit
    
    n = len(ready_operations)
    
    # Extract features for all ready operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    # For each operation, find the minimum duration across machine options
    min_durations = np.array([min(op["machine_options"].values()) for op in ready_operations])
    max_durations = np.array([max(op["machine_options"].values()) for op in ready_operations])
    
    # Normalize features to [0, 1] range
    rw_min, rw_max = remaining_works.min(), remaining_works.max()
    rd_min, rd_max = min_durations.min(), min_durations.max()
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    
    def _norm(x, x_min, x_max):
        if x_max == x_min:
            return np.ones_like(x) * 0.5
        return (x - x_min) / (x_max - x_min)
    
    rw_norm = _norm(remaining_works, rw_min, rw_max)
    rd_norm = _norm(min_durations, rd_min, rd_max)
    ro_norm = _norm(remaining_ops, ro_min, ro_max)
    
    # Composite score:
    # - Higher remaining work is better (urgency)
    # - Lower min duration is better (less blocking)
    # - Lower remaining ops is slightly better (closer to completion)
    # - Penalize operations with very long max durations
    
    max_d_norm = _norm(max_durations, max_durations.min(), max_durations.max())
    
    # Weights tuned for open-shop scheduling
    w_rw = 0.45    # Weight for remaining work (urgency)
    w_rd = 0.30    # Weight for min duration (efficiency)
    w_ro = 0.10    # Weight for remaining ops (completion proximity)
    w_md = 0.15    # Weight for max duration penalty
    
    # Score: higher is better
    scores = (
        w_rw * rw_norm +
        w_rd * (1.0 - rd_norm) +  # Prefer shorter min durations
        w_ro * (1.0 - ro_norm) +  # Prefer fewer remaining ops
        w_md * (1.0 - max_d_norm) # Prefer shorter max durations
    )
    
    # Check deadline
    if _time.time() > _deadline:
        # Fallback: just pick the one with most remaining work
        return int(np.argmax(remaining_works))
    
    # Return index of operation with highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
