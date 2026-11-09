"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def _soft_min(values, alpha=2.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation."""
    values = np.asarray(values, dtype=np.float64)
    # Use negative log-sum-exp of negative values
    neg_vals = -values
    max_neg = np.max(neg_vals)
    if max_neg == 0:
        return np.exp(neg_vals) / np.sum(np.exp(neg_vals))
    shifted = neg_vals - max_neg
    log_sum = max_neg + np.log(np.sum(np.exp(shifted)))
    return np.exp(-log_sum)

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Returns index of the ready operation to start next.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, collect all machine durations
    all_durations = []
    for op in ready_operations:
        if op["machine_options"]:
            all_durations.append(list(op["machine_options"].values()))
        else:
            all_durations.append([1.0])
    
    # Compute min, mean, and soft-min durations
    min_durations = np.array([min(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    mean_durations = np.array([np.mean(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    
    # Use soft-min with alpha=2 for smooth approximation
    soft_min_durations = np.array([_soft_min(np.array(d, dtype=np.float64), alpha=2.0) if d else 1.0 
                                    for d in all_durations], dtype=np.float64)

    # Normalize features
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd_min = _normalize(min_durations)
    nd_mean = _normalize(mean_durations)
    nd_soft = _normalize(soft_min_durations)

    # Compute average work per remaining op
    avg_work_per_op = remaining_work / np.maximum(remaining_ops, 1.0)
    naw = _normalize(avg_work_per_op)

    # Weights: balance urgency, flow, SPT, and work-per-op
    # Adjusted weights based on findings: emphasize SPT and urgency
    w_work = 0.30      # urgency (higher remaining work)
    w_ops = 0.20       # keep in motion (higher remaining ops)
    w_spt = 0.35       # shortest processing time (lower duration is better)
    w_avg = 0.15       # average work per op

    # Score: maximize
    # - remaining_work (urgency): higher is better
    # - remaining_ops (flow): higher is better  
    # - (1 - nd_soft) for SPT: lower soft-min duration is better
    # - (1 - naw): lower average work per op is better (simpler ops)
    
    score = (w_work * nw + 
             w_ops * no + 
             w_spt * (1.0 - nd_soft) + 
             w_avg * (1.0 - naw))

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
