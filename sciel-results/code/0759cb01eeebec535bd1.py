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
    
    Strategy: Combine multiple heuristics to minimize makespan:
    1. Most remaining work (MRW) - prioritize jobs with more work left
    2. Smooth LPT via soft-min/soft-max of machine durations
    3. Remaining operations count - keep pipeline full
    4. Time-aware urgency: operations with less slack relative to current time
    5. Downstream unlock: prefer ops that free up more remaining ops
    
    We use a weighted combination with smooth surrogates for robustness.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Smooth soft-min and soft-max of machine option durations via log-sum-exp
    # soft-min(x) = -logsumexp(-x)/len  approximates min
    # soft-max(x) = logsumexp(x)/len   approximates max
    def soft_min(durations, beta=0.1):
        """Smooth approximation of min using log-sum-exp."""
        if not durations:
            return 0.0
        d = np.array(durations, dtype=np.float64)
        if len(d) == 1:
            return float(d[0])
        # soft min = - (1/beta) * log(sum(exp(-beta * d)))
        neg_beta_d = -beta * d
        max_val = np.max(neg_beta_d)
        log_sum = max_val + np.log(np.sum(np.exp(neg_beta_d - max_val)))
        return float(-log_sum / beta)
    
    def soft_max(durations, beta=0.1):
        """Smooth approximation of max using log-sum-exp."""
        if not durations:
            return 0.0
        d = np.array(durations, dtype=np.float64)
        if len(d) == 1:
            return float(d[0])
        beta_d = beta * d
        max_val = np.max(beta_d)
        log_sum = max_val + np.log(np.sum(np.exp(beta_d - max_val)))
        return float(log_sum / beta)
    
    min_durations = np.array([
        soft_min(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    max_durations = np.array([
        soft_max(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Average processing time
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Spread: range of machine options (indicates flexibility)
    spread = max_durations - min_durations
    
    # Normalize features
    def normalize(x):
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    nw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    md_norm = normalize(min_durations)  # Higher is longer duration
    ad_norm = normalize(avg_durations)
    mx_norm = normalize(max_durations)
    sp_norm = normalize(spread)
    
    # Time-aware urgency: how much time has elapsed relative to total work
    if time > 0:
        safe_time = max(time, 1e-6)
        urgency = remaining_work / safe_time
        urgency_norm = normalize(urgency)
    else:
        urgency_norm = nw_norm
    
    # Downstream unlock: remaining_ops / (remaining_work + eps)
    # Higher means each unit of work unlocks more future ops
    eps = 1e-6
    unlock_ratio = remaining_ops / (remaining_work + eps)
    unlock_norm = normalize(unlock_ratio)
    
    # Composite score: we want to maximize this
    # Hypothesis: LPT (Longest Processing Time) helps balance loads early.
    # We prioritize long operations (high min_duration), jobs with high remaining work,
    # and operations that unlock more downstream work.
    
    w_rw = 0.35   # weight for remaining work
    w_md = 0.30   # weight for soft-min duration (positive: LPT)
    w_ro = 0.10   # weight for remaining ops
    w_urg = 0.10  # weight for time-aware urgency
    w_unlock = 0.10  # weight for downstream unlock
    w_spread = 0.05  # small weight for flexibility (spread)
    
    score = (w_rw * nw_norm + 
             w_md * md_norm + 
             w_ro * ro_norm +
             w_urg * urgency_norm +
             w_unlock * unlock_norm +
             w_spread * sp_norm)
    
    # Find the best index
    idx = int(np.argmax(score))
    
    # Smooth tie-breaking: if scores are very close, use a secondary smooth score
    max_score = score.max()
    close = np.where(score > max_score - 1e-6)[0]
    if len(close) > 1:
        # Among close candidates, use a smooth secondary score:
        # prefer longer soft-max duration and more remaining work
        close_md = min_durations[close]
        close_mx = max_durations[close]
        close_rw = remaining_work[close]
        
        def normalize_local(x):
            rng = x.max() - x.min()
            if rng > 1e-12:
                return (x - x.min()) / rng
            return np.zeros_like(x)
        
        sec_score = (0.4 * normalize_local(close_md) + 
                     0.3 * normalize_local(close_mx) + 
                     0.3 * normalize_local(close_rw))
        best_local = int(np.argmax(sec_score))
        idx = int(close[best_local])
    
    return int(idx)
# EVOLVE-BLOCK-END
