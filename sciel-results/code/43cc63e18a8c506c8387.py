import numpy as np
import time as _time_module

# EVOLVE-BLOCK-START
_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for Job-Shop Scheduling.
    
    Combines the core WSPT-like heuristic (remaining_ops / min_duration) from the
    current program with a deadline guard and a composite refinement that also
    considers remaining_work to handle critical-path jobs.
    
    Primary score: remaining_ops / min_duration (maximize)
    Tie-breaker / refinement: incorporate remaining_work to favor jobs with
    less total work remaining when primary scores are close.
    """
    if not ready_operations:
        return 0

    # Deadline guard: if we're getting close to the limit, use fast fallback
    if _time_module.perf_counter() > _DEADLINE:
        # Fast fallback: just pick the one with most remaining work (seed behavior)
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)

    # Compute metrics for each ready operation
    soft_min_durations = []
    avg_durations = []
    remaining_works = []
    remaining_ops_list = []

    # Temperature for soft-min: lower = closer to hard min, higher = smoother
    # Using 0.1 gives a smooth approximation that still heavily weights the minimum
    beta = 10.0  # 1/temperature

    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        if not machine_options:
            soft_min_dur = 1.0
            avg_dur = 1.0
        else:
            durations = np.array(list(machine_options.values()), dtype=np.float64)
            # Soft-min via log-sum-exp: -log(sum(exp(-beta*d))) / beta
            # This is smooth and differentiable, approximating min as beta->inf
            max_d = np.max(durations)
            log_sum = np.log(np.sum(np.exp(-beta * (durations - max_d))))
            soft_min_dur = max_d - log_sum / beta
            avg_dur = float(np.mean(durations))
            if soft_min_dur <= 0:
                soft_min_dur = 1e-9
            if avg_dur <= 0:
                avg_dur = 1e-9

        soft_min_durations.append(soft_min_dur)
        avg_durations.append(avg_dur)

        rem_work = op.get("remaining_work", 0)
        if rem_work <= 0:
            rem_work = soft_min_dur
        remaining_works.append(rem_work)

        rem_ops = op.get("remaining_ops", 1)
        if rem_ops <= 0:
            rem_ops = 1
        remaining_ops_list.append(rem_ops)

    # Convert to numpy arrays for vectorized computation
    soft_min_durations = np.array(soft_min_durations, dtype=np.float64)
    avg_durations = np.array(avg_durations, dtype=np.float64)
    remaining_works = np.array(remaining_works, dtype=np.float64)
    remaining_ops_arr = np.array(remaining_ops_list, dtype=np.float64)

    # Primary score: remaining_ops / soft_min_duration (smooth WSPT-like, maximize)
    # Soft-min is slightly larger than hard min, making scores slightly lower
    # but smoother, which can improve tie-breaking behavior
    primary_score = remaining_ops_arr / soft_min_durations

    # Refinement: incorporate remaining_work as a secondary signal
    # Normalize remaining_work to [0,1] (lower is better for makespan)
    def normalize_lower_better(arr):
        """Normalize so that lower values get higher scores in [0,1]."""
        if len(arr) == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        normed = 1.0 - (arr - mn) / (mx - mn)
        return normed

    # Normalize primary_score to [0,1]
    p_min, p_max = primary_score.min(), primary_score.max()
    if p_max - p_min < 1e-12:
        primary_norm = np.ones_like(primary_score) * 0.5
    else:
        primary_norm = (primary_score - p_min) / (p_max - p_min)

    # Normalize remaining_work (lower remaining work is better)
    work_norm = normalize_lower_better(remaining_works)

    # Weighted combination:
    # - Primary WSPT score gets most weight (it drove the 1.0583 improvement)
    # - Remaining work gets a moderate weight as a refinement signal
    # The avg_duration term adds slight machine-load awareness
    avg_norm = normalize_lower_better(avg_durations)

    # Slightly increase weight on remaining_work (better critical-path signal)
    # and reduce primary weight marginally to allow more discrimination
    weights = np.array([0.55, 0.30, 0.15])
    total_score = weights[0] * primary_norm + weights[1] * work_norm + weights[2] * avg_norm

    # Return index of operation with highest priority score
    return int(np.argmax(total_score))
# EVOLVE-BLOCK-END
