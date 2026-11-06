"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Hard deadline guard: keep a safety margin before the 120s limit.
_DEADLINE_START = None
_DEADLINE_LIMIT = 110.0  # seconds


def _now():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = _time.perf_counter()
    return _time.perf_counter() - _DEADLINE_START


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    Returns the index of the ready operation to start next.

    Strategy:
      - Combine multiple heuristics to approximate critical-path priority.
      - Weights:
          * remaining_work (total remaining processing time for the job)
          * remaining_ops (number of operations left in the job)
          * min_duration (shortest processing time among machine options)
          * max_duration (longest processing time among machine options)
      - The idea: prioritize jobs with more remaining work (critical path),
        but also consider operation count and duration spread to balance
        machine utilization.
    """
    # Deadline guard: if we've exceeded the limit, fall back to a simple rule.
    if _now() > _DEADLINE_LIMIT:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Precompute features for each ready operation.
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # For each op, get min and max duration from machine_options.
    min_durs = np.zeros(n, dtype=float)
    max_durs = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        if durations:
            min_durs[i] = min(durations)
            max_durs[i] = max(durations)

    # Normalize features to [0, 1] range for fair weighting.
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    nw_norm = normalize(remaining_work)
    no_norm = normalize(remaining_ops)
    md_norm = normalize(min_durs)
    xd_norm = normalize(max_durs)

    # Weighted score: higher is more urgent/prioritized.
    # Emphasize remaining work most, then avg work per op, then duration characteristics.
    # Use soft-min (log-sum-exp) for a smooth surrogate of min_duration.
    # Lower soft-min duration -> higher priority.
    # Small penalty for jobs with many remaining ops to avoid over-concentration.
    w_work = 0.55
    w_avg = 0.25
    w_mindur = 0.15
    w_ops_penalty = 0.05

    # Compute soft-min of durations for each operation (temperature-based)
    # soft_min = -log(sum(exp(-d_i / T))) * T, which approximates min as T -> 0
    T = 0.5  # lower temperature -> sharper, closer to true min
    soft_min_durs = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        durations = np.array(list(op["machine_options"].values()), dtype=float)
        if len(durations) > 0:
            # Use log-sum-exp for numerical stability
            shifted = -durations / T
            max_shifted = np.max(shifted)
            log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
            soft_min_durs[i] = -log_sum_exp * T
        else:
            soft_min_durs[i] = 0.0

    # Normalize soft_min_durs to [0, 1]
    smd_norm = normalize(soft_min_durs)

    # Compute avg remaining work per operation
    # Guard against division by zero
    avg_rem_per_op = np.zeros(n, dtype=float)
    for i in range(n):
        if remaining_ops[i] > 1e-12:
            avg_rem_per_op[i] = remaining_work[i] / remaining_ops[i]
        else:
            avg_rem_per_op[i] = remaining_work[i]

    arpo_norm = normalize(avg_rem_per_op)

    # Compute slack-based priority: for each operation, find the minimum
    # (earliest available time on best machine - current time) across all
    # machine options. Lower slack = more urgent to start.
    # This captures machine availability constraints that pure work-based
    # heuristics miss.
    
    # For each operation, compute the minimum "slack" = min over machines of
    # (machine_earliest_available - current_time). We don't have direct access
    # to machine availability, but we can approximate: operations with shorter
    # durations on some machine have more scheduling flexibility (higher slack),
    # while operations where all machines have long durations are more constrained.
    # 
    # Alternative: use the range of durations as a proxy for constraint tightness.
    # Operations with narrow duration ranges (all machines similar) are more
    # constrained in machine choice. Operations with wide ranges have more options.
    dur_range = max_durs - min_durs
    dr_norm = normalize(dur_range)
    
    # Also compute a "criticality" based on how many machines have short durations
    # (i.e., how many machines can handle this op quickly)
    short_dur_count = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        if durations:
            threshold = np.mean(durations)
            short_dur_count[i] = sum(1 for d in durations if d <= threshold)
    
    sdc_norm = normalize(short_dur_count)
    
    # Updated weights: add slack/constraint signals
    w_work = 0.45
    w_avg = 0.20
    w_mindur = 0.15
    w_ops_penalty = 0.05
    w_range = 0.10      # narrow range = more constrained = higher priority
    w_short_count = 0.05  # more short-duration machines = more flexible = lower priority
    
    # Score: higher is more prioritized
    # - High remaining work -> urgent
    # - High avg work per op -> each remaining op is substantial (critical path)
    # - Short soft-min duration -> can start quickly, frees machine sooner
    # - Small penalty for many remaining ops (spread risk)
    # - Narrow duration range -> constrained machine choice -> higher priority
    # - Fewer short-duration machines -> less flexible -> higher priority
    score = (w_work * nw_norm +
             w_avg * arpo_norm +
             w_mindur * (1.0 - smd_norm) -
             w_ops_penalty * no_norm +
             w_range * (1.0 - dr_norm) +
             w_short_count * (1.0 - sdc_norm))

    return int(np.argmax(score))
# EVOLVE-BLOCK-END
