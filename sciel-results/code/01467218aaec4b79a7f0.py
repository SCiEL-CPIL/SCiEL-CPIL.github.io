"""Dispatch rule for open-shop scheduling: choose ready operation with best priority."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: track start time for safety
_START_TIME = _time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def _elapsed_ok():
    """Check if we're still within safe time budget."""
    return (_time.time() - _START_TIME) < (150.0 - _DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.
    
    Strategy: LPT-inspired greedy for makespan minimization in open-shop.
    Prioritize operations with the longest minimum processing time to
    balance machine loads early and prevent late-stage bottlenecks.
    Combine with remaining work urgency for a robust composite score.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Minimum processing time among machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Maximum processing time among machine options
    max_durations = np.array([
        max(op["machine_options"].values()) if op["machine_options"] else 0.0
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
    
    md_norm = normalize(min_durations)
    maxd_norm = normalize(max_durations)
    ad_norm = normalize(avg_durations)
    nw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    
    # Time-aware urgency
    if time > 0:
        safe_time = max(time, 1e-6)
        urgency = remaining_work / safe_time
        urgency_norm = normalize(urgency)
    else:
        urgency_norm = nw_norm
    
    # LPT-inspired composite score: prioritize long minimum processing times
    # - High min duration: good (LPT, balance loads early)
    # - High max duration: good (critical path)
    # - High remaining work: good (finish big jobs)
    # - High urgency: good (time-critical)
    # - Low remaining ops: slight preference (fewer ops left = closer to done)
    
    w_md = 0.35   # weight for min duration (LPT core)
    w_maxd = 0.20 # weight for max duration
    w_rw = 0.20   # weight for remaining work
    w_urg = 0.15  # weight for urgency
    w_ro = 0.10   # weight for remaining ops (inverted)
    
    score = (w_md * md_norm + 
             w_maxd * maxd_norm + 
             w_rw * nw_norm + 
             w_urg * urgency_norm +
             w_ro * (1.0 - ro_norm))
    
    # Find the best index
    idx = int(np.argmax(score))
    
    # Smooth tie-breaking using soft-max via log-sum-exp on min_durations
    # Prefer operations with longer minimum durations in case of ties
    max_score = score.max()
    close = np.where(score > max_score - 1e-4)[0]
    if len(close) > 1:
        # Soft-max: use log-sum-exp with positive durations to get smooth maximum
        # We want to prefer longer durations, so use +durations in log-sum-exp
        durations = min_durations[close]
        # Temperature for softness
        beta = 10.0
        # Log-sum-exp for numerical stability
        max_dur = np.max(durations)
        contributions = np.exp(beta * (durations - max_dur))
        # Normalize to get probabilities
        probs = contributions / contributions.sum()
        # Pick the index with highest probability (longest duration in soft sense)
        idx = close[int(np.argmax(probs))]
    
    return int(idx)
# EVOLVE-BLOCK-END
