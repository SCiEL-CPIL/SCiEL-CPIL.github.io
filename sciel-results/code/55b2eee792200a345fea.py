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


def _soft_min(durations, beta=10.0):
    """
    Smooth surrogate for min using log-sum-exp with negative sign.
    soft_min(x) = -log(sum(exp(-beta * x))) / beta
    As beta -> inf, this approaches true min.
    Higher beta = sharper (closer to hard min).
    """
    if len(durations) == 0:
        return 0.0
    if len(durations) == 1:
        return float(durations[0])
    x = -beta * durations
    x_max = np.max(x)
    if x_max > 100:
        return float(np.min(durations))
    exp_sum = np.sum(np.exp(x - x_max))
    if exp_sum <= 0 or not np.isfinite(exp_sum):
        return float(np.min(durations))
    return float(-np.log(exp_sum) / beta - x_max / beta)


def _soft_spread(durations, beta=5.0):
    """
    Measure how 'tight' the durations are around the min.
    A small spread means all machines are similarly fast -> good for load balance.
    A large spread means one machine is much faster -> less flexibility benefit.
    Returns a value in [0, 1] where 1 = all equal, 0 = very spread out.
    """
    if len(durations) <= 1:
        return 1.0
    d = np.array(durations, dtype=np.float64)
    mn = np.min(d)
    mx = np.max(d)
    rng = mx - mn
    if rng < 1e-12:
        return 1.0
    # Normalize spread: 1 - (spread / max_duration)
    # Use a soft measure: exp(-beta * spread / mean)
    mean_d = np.mean(d)
    if mean_d < 1e-12:
        return 1.0
    spread_ratio = rng / mean_d
    return float(np.exp(-beta * spread_ratio))


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.
    
    Strategy (refined composite heuristic):
    1. Urgency: average remaining work per remaining op (jobs close to completion
       have higher per-op urgency, reducing makespan).
    2. Duration: soft-min over machine options (shorter effective ops free machines sooner).
    3. Flexibility/spread: how tightly clustered the machine durations are
       (tight cluster = better load-balancing potential).
    4. Machine selection: shortest duration machine.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Deadline check: if we're running out of time, use fast greedy
    if not _elapsed_ok():
        # Fast greedy: pick op with smallest remaining_work
        min_rw = float('inf')
        min_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op["remaining_work"]
            if rw < min_rw:
                min_rw = rw
                min_idx = i
        return min_idx

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Avoid division by zero
    remaining_ops_safe = np.where(remaining_ops > 0, remaining_ops, 1.0)
    
    # Urgency: average remaining work per remaining op
    # Jobs with fewer remaining ops but still significant work are more urgent
    avg_rw_per_op = remaining_work / remaining_ops_safe
    
    soft_min_durations = np.zeros(n, dtype=np.float64)
    true_min_durations = np.zeros(n, dtype=np.float64)
    spreads = np.zeros(n, dtype=np.float64)
    n_options = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if opts:
            durations = np.array(list(opts.values()), dtype=np.float64)
            soft_min_durations[i] = _soft_min(durations, beta=10.0)
            true_min_durations[i] = np.min(durations)
            spreads[i] = _soft_spread(durations, beta=5.0)
            n_options[i] = len(durations)
        else:
            soft_min_durations[i] = 0.0
            true_min_durations[i] = 0.0
            spreads[i] = 1.0
            n_options[i] = 0.0
    
    # Normalize features
    def normalize(x):
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)
    
    urgency_norm = normalize(avg_rw_per_op)
    sdm_norm = normalize(soft_min_durations)
    spread_norm = normalize(spreads)
    flex_norm = normalize(n_options)
    
    # Composite score - maximize:
    # - Urgency (high avg remaining work per op = critical path)
    # - Inverse soft-min duration (shorter ops free machines sooner)
    # - Spread tightness (tight clusters = better load balance)
    # - Flexibility (more options)
    
    w_urgency = 0.45
    w_duration = 0.30
    w_spread = 0.15
    w_flex = 0.10
    
    score = (w_urgency * urgency_norm + 
             w_duration * (1.0 - sdm_norm) + 
             w_spread * spread_norm + 
             w_flex * flex_norm)
    
    # Find the best index
    idx = int(np.argmax(score))
    
    # Tie-breaking: if scores are close, prefer higher urgency, then shorter soft-min
    max_score = score.max()
    close = np.where(score > max_score - 1e-4)[0]
    if len(close) > 1:
        close_urgency = avg_rw_per_op[close]
        close_sdm = soft_min_durations[close]
        
        def normalize_local(x):
            rng = x.max() - x.min()
            if rng > 1e-12:
                return (x - x.min()) / rng
            return np.zeros_like(x)
        
        sec_score = (0.6 * normalize_local(close_urgency) + 
                     0.4 * (1.0 - normalize_local(close_sdm)))
        best_local = int(np.argmax(sec_score))
        idx = int(close[best_local])
    
    # Select the best machine for the chosen operation
    # Pick the machine with the shortest duration to minimize machine occupation
    op = ready_operations[idx]
    machine_to_use = None
    min_dur = float('inf')
    
    for machine, duration in op["machine_options"].items():
        if duration < min_dur:
            min_dur = duration
            machine_to_use = machine
    
    return (idx, machine_to_use)

# EVOLVE-BLOCK-END
