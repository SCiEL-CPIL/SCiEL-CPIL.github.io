import numpy as np
import time as _time_module

# EVOLVE-BLOCK-START
_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def _soft_min(values, temperature=1.0):
    """
    Soft-min via log-sum-exp: -T * log(sum(exp(-v/T)))
    As T -> 0, approaches true min. For T=1, gives a smooth surrogate.
    """
    if len(values) == 0:
        return 1.0
    v = np.asarray(values, dtype=np.float64)
    v = np.maximum(v, 1e-9)  # avoid log(0) or log(negative)
    # Soft min: -T * log(sum(exp(-v/T)))
    # For numerical stability, subtract max of -v/T = -min(v)/T
    neg_v_over_t = -v / temperature
    max_neg = np.max(neg_v_over_t)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_v_over_t - max_neg)))
    return -temperature * log_sum


def dispatch(ready_operations, time):
    """
    Dispatch rule for Job-Shop Scheduling.
    
    Composite weighted score combining:
    1. WSPT-like urgency: remaining_ops / soft_min_duration (maximize)
    2. Machine-contention avoidance: min_dur / sum_durations ratio (maximize)
       High ratio = one machine clearly best = less contention
    3. Remaining work (lower is better)
    4. Duration variance penalty (lower is better, less machine choice conflict)
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're getting close to the limit, use fast fallback
    if _time_module.perf_counter() > _DEADLINE:
        # Fast fallback: just pick the one with most remaining work (seed behavior)
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)

    # Compute metrics for each ready operation
    min_durations = []
    sum_durations = []
    var_durations = []
    remaining_works = []
    remaining_ops_list = []
    soft_min_durations = []

    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        if not machine_options:
            min_dur = 1.0
            sum_dur = 1.0
            var_dur = 0.0
            soft_min_dur = 1.0
        else:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            durations = np.maximum(durations, 1e-9)
            min_dur = float(np.min(durations))
            sum_dur = float(np.sum(durations))
            var_dur = float(np.var(durations))
            # Soft-min with temperature 0.3 for a tighter approximation to min
            soft_min_dur = float(_soft_min(durations, temperature=0.3))

        min_durations.append(min_dur)
        sum_durations.append(sum_dur)
        var_durations.append(var_dur)
        soft_min_durations.append(soft_min_dur)

        rem_work = op.get("remaining_work", 0)
        if rem_work <= 0:
            rem_work = min_dur
        remaining_works.append(rem_work)

        rem_ops = op.get("remaining_ops", 1)
        if rem_ops <= 0:
            rem_ops = 1
        remaining_ops_list.append(rem_ops)

    # Convert to numpy arrays for vectorized computation
    min_durations = np.array(min_durations, dtype=np.float64)
    sum_durations = np.array(sum_durations, dtype=np.float64)
    var_durations = np.array(var_durations, dtype=np.float64)
    remaining_works = np.array(remaining_works, dtype=np.float64)
    remaining_ops_arr = np.array(remaining_ops_list, dtype=np.float64)
    soft_min_durations = np.array(soft_min_durations, dtype=np.float64)

    # Primary score: remaining_ops / soft_min_duration (WSPT-like, maximize)
    # Using soft-min with lower temperature for tighter min approximation
    primary_score = remaining_ops_arr / np.maximum(soft_min_durations, 1e-9)

    # Machine-contention avoidance: min_dur / sum_durations (maximize)
    # High ratio means one machine is clearly the best choice (less spread)
    # This indicates less contention with other operations on that machine
    contention_ratio = min_durations / np.maximum(sum_durations, 1e-9)

    def normalize_lower_better(arr):
        """Normalize so that lower values get higher scores in [0,1]."""
        if len(arr) == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        normed = 1.0 - (arr - mn) / (mx - mn)
        return normed

    def normalize_higher_better(arr):
        """Normalize so that higher values get higher scores in [0,1]."""
        if len(arr) == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        normed = (arr - mn) / (mx - mn)
        return normed

    # Normalize primary_score to [0,1] (higher is better)
    primary_norm = normalize_higher_better(primary_score)

    # Normalize contention ratio (higher is better = less contention)
    contention_norm = normalize_higher_better(contention_ratio)

    # Normalize remaining_work (lower remaining work is better)
    work_norm = normalize_lower_better(remaining_works)

    # Normalize duration variance (lower is better = less machine choice conflict)
    var_norm = normalize_lower_better(var_durations)

    # Weighted combination:
    # - Primary WSPT score: most important (drives urgency) - increased weight
    # - Contention ratio: machine-contention avoidance
    # - Remaining work: tie-breaker for jobs with less total work
    # - Duration variance: penalize operations with very different machine durations
    weights = np.array([0.50, 0.22, 0.15, 0.13])
    total_score = (weights[0] * primary_norm + 
                   weights[1] * contention_norm + 
                   weights[2] * work_norm + 
                   weights[3] * var_norm)

    # Return index of operation with highest priority score
    return int(np.argmax(total_score))
# EVOLVE-BLOCK-END
