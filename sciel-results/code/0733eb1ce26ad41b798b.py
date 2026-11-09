"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level deadline guard: we must finish within 120s per evaluation.
# The dispatch function is called many times per instance; we track total
# elapsed time across calls to ensure we don't exceed the budget.
_START_TIME = _time.monotonic()
TIME_BUDGET = 115.0  # safety margin before 120s limit


def _elapsed():
    return _time.monotonic() - _START_TIME


def _soft_min(values, temperature=1.0):
    """
    Compute soft-min of an array using log-sum-exp trick.
    soft_min = -logsumexp(-values / temperature) * temperature
    As temperature -> 0, approaches true min.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    scaled = -values / temperature
    max_scaled = scaled.max()
    if max_scaled > 700:  # avoid overflow
        return values.min()
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum_exp * temperature


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: Ratio-based scoring that balances urgency (remaining work) with
    processing time efficiency. Uses soft-min for smooth machine option evaluation
    and a critical path bonus for jobs on their critical path.
    
    Core idea: score = remaining_work / (1 + effective_processing_time)
    where effective_processing_time uses soft-min over machine options.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Check time budget - if we're running low, fall back to fast MWL
    if _elapsed() > TIME_BUDGET - 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Extract features
    rw = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    ro = np.array([max(op["remaining_ops"], 1) for op in ready_operations], dtype=np.float64)
    
    # Compute soft-min over machine options for each operation
    soft_min_durs = np.zeros(n)
    best_durs = np.zeros(n)
    n_opts = np.zeros(n)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durs = np.array(list(mo.values()), dtype=np.float64)
            soft_min_durs[i] = _soft_min(durs, temperature=1.0)
            best_durs[i] = durs.min()
            n_opts[i] = len(durs)
        else:
            soft_min_durs[i] = 0.0
            best_durs[i] = 0.0
            n_opts[i] = 0.0
    
    # Normalize remaining work
    rw_max = rw.max() if rw.max() > 0 else 1.0
    
    # Core ratio score: urgency / processing_time_efficiency
    # Higher remaining work relative to processing time is better
    # Use 1 + soft_min to avoid division by zero and smooth the objective
    effective_dur = 1.0 + soft_min_durs
    ratio_scores = rw / effective_dur
    
    # Normalize ratio scores
    ratio_max = ratio_scores.max() if ratio_scores.max() > 0 else 1.0
    normalized_ratio = ratio_scores / ratio_max
    
    # Critical path proxy: average remaining operation duration
    avg_dur = rw / ro
    avg_max = avg_dur.max() if avg_dur.max() > 0 else 1.0
    normalized_avg = avg_dur / avg_max
    
    # Flexibility bonus: more machine options is slightly better
    no_max = n_opts.max() if n_opts.max() > 0 else 1.0
    flexibility = n_opts / no_max
    
    # Combine: primary ratio score + secondary critical path + flexibility
    # Weight: 0.7 for ratio, 0.2 for critical path, 0.1 for flexibility
    scores = 0.7 * normalized_ratio + 0.2 * normalized_avg + 0.1 * flexibility
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
