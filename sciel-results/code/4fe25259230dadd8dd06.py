"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


def _elapsed_ok():
    """Check if we're still within the deadline with safety margin."""
    return (_time.time() - _START_TIME) < (_DEADLINE - _SAFETY_MARGIN)


def _soft_max(durations, temperature=0.1):
    """Smooth surrogate for max using log-sum-exp.
    
    Returns a value close to max(durations) but smooth/differentiable.
    Lower temperature -> closer to hard max.
    """
    if not durations:
        return 0.0
    arr = np.asarray(durations, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    
    # Use log-sum-exp for numerical stability
    a = arr / temperature
    a_max = np.max(a)
    # Soft max ≈ temperature * (max(a) + log(mean(exp(a - max(a)))))
    # But we want something between min and max, weighted toward max
    # Using: temp * (logsumexp(a) - log(n)) gives a smooth mean
    # For a "soft max" that's closer to max: use larger temperature sensitivity
    # Actually, let's use a weighted combination: 0.7*max + 0.3*mean as a smooth surrogate
    # that captures both the best and average machine availability
    
    m = float(np.max(arr))
    mean_val = float(np.mean(arr))
    # Smooth interpolation: as temperature -> 0, this approaches max
    # Using exponential weighting
    exp_a = np.exp(a - a_max)
    weights = exp_a / np.sum(exp_a)
    soft_max_val = float(np.sum(weights * arr))
    
    return soft_max_val


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard: if we're running out of time, return quickly
    if not _elapsed_ok():
        # Fallback: pick the operation with most remaining work (simple heuristic)
        best_idx = 0
        best_work = -1.0
        for i, op in enumerate(ready_operations):
            rw = float(op.get("remaining_work", 0.0))
            if rw > best_work:
                best_work = rw
                best_idx = i
        return int(best_idx)
    
    scores = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        rem_work = float(op.get("remaining_work", 0.0))
        rem_ops = max(int(op.get("remaining_ops", 1)), 1)
        
        # Use soft-max over machine durations as a smooth surrogate
        machine_opts = op.get("machine_options", {})
        if machine_opts:
            durations = list(machine_opts.values())
            # Soft max captures both the best machine and the distribution
            dur_score = _soft_max(durations, temperature=0.15)
            min_dur = min(durations)
            max_dur = max(durations)
            # Combine: weight toward max (best case) but informed by spread
            dur_combined = 0.6 * dur_score + 0.4 * min_dur
        else:
            dur_score = 0.0
            min_dur = 0.0
            dur_combined = 0.0
        
        # Composite score:
        # 1. Remaining work on job (urgency)
        # 2. Duration score (smooth surrogate for machine utilization)
        # 3. Average remaining work per op (balance)
        avg_rem = rem_work / rem_ops
        
        score = 1.0 * rem_work + 0.9 * dur_combined + 0.4 * avg_rem
        scores[i] = score
    
    # Tie-breaking: prefer operations with fewer remaining ops (closer to completion)
    for i, op in enumerate(ready_operations):
        rem_ops = max(int(op.get("remaining_ops", 1)), 1)
        scores[i] -= 0.03 * rem_ops
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
