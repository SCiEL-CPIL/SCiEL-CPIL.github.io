"""Dispatch rule for open-shop scheduling: choose ready operation with best priority."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.
    
    Strategy: Combine multiple heuristics to minimize makespan:
    1. Most remaining work (MRW) - the original seed
    2. Minimum processing time (SPT) - reduce idle time
    3. Balance across machines - prefer machines with less current load
    4. Consider remaining operations count - prioritize jobs with more ops left
    
    We use a weighted combination that adapts based on the current state.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Minimum processing time among machine options (SPT heuristic)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Average processing time
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Number of machine options (flexibility)
    n_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    
    # Normalize features
    def normalize(x):
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    nw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    md_norm = normalize(min_durations)  # lower is better for SPT
    ad_norm = normalize(avg_durations)
    
    # Composite score: we want to maximize this
    # - High remaining work: good (finish big jobs early)
    # - Low min duration: good (SPT, less idle time)
    # - High remaining ops: good (keep pipeline full)
    # - Low avg duration: good (faster completion)
    
    w_rw = 0.35  # weight for remaining work
    w_md = 0.25  # weight for min duration (inverted)
    w_ro = 0.25  # weight for remaining ops
    w_ad = 0.15  # weight for avg duration (inverted)
    
    score = (w_rw * nw_norm + 
             w_md * (1.0 - md_norm) + 
             w_ro * ro_norm + 
             w_ad * (1.0 - ad_norm))
    
    # Break ties by preferring shorter min duration
    idx = int(np.argmax(score))
    
    # If scores are very close, use SPT as tiebreaker
    max_score = score.max()
    close = np.where(score > max_score - 1e-6)[0]
    if len(close) > 1:
        close_durations = min_durations[close]
        idx = close[np.argmin(close_durations)]
    
    return int(idx)
# EVOLVE-BLOCK-END
