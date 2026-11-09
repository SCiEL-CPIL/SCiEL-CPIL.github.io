"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_DEADLINE = _time.time() + 145.0  # 5s safety margin
_DEADLINE_CHECK_INTERVAL = 0.05   # seconds between checks


def _deadline_exceeded():
    return _time.time() >= _DEADLINE


def _soft_min(values, temperature=1.0):
    """
    Soft-min via log-sum-exp: -temperature * log(sum(exp(-v/temperature)))
    This is a smooth surrogate for min(values) that captures the full distribution.
    As temperature -> 0, it approaches the true min.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    # Numerically stable log-sum-exp
    shifted = np.asarray(values, dtype=np.float64) / temperature
    max_shifted = np.max(shifted)
    if not np.isfinite(max_shifted):
        return min(values)
    exp_shifted = np.exp(shifted - max_shifted)
    log_sum_exp = max_shifted + np.log(np.sum(exp_shifted))
    return -temperature * log_sum_exp


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Composite scoring that blends:
    1. Remaining work (job completion priority)
    2. Soft-min of machine durations (smooth surrogate for min duration, captures flexibility)
    3. Duration spread (penalizes operations where best option is much better than others - 
       indicates potential contention if we pick the wrong machine)
    4. Number of remaining ops (job urgency)
    5. Work density (remaining_work / remaining_ops) - identifies bottleneck jobs
    
    The score is a weighted composite. We pick the operation with the highest score.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Soft-min of machine durations: smooth surrogate for min, captures distribution
    soft_min_durs = np.array([
        _soft_min(list(op["machine_options"].values()), temperature=0.5)
        for op in ready_operations
    ], dtype=np.float64)

    # Duration spread: how much the best option differs from others
    # Lower spread means more flexibility (less risk of contention)
    duration_spreads = np.array([
        (max(op["machine_options"].values()) - min(op["machine_options"].values())) 
        if len(op["machine_options"]) > 1 else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Work density: remaining_work / max(remaining_ops, 1)
    safe_ops = np.maximum(remaining_ops, 1.0)
    work_density = remaining_works / safe_ops

    # Normalize each feature to [0, 1] range
    def _normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    norm_work = _normalize(remaining_works)
    norm_soft_dur = _normalize(soft_min_durs)
    norm_spread = _normalize(duration_spreads)
    norm_ops = _normalize(remaining_ops)
    norm_density = _normalize(work_density)

    # Weights: prioritize remaining work, then soft-min duration (lower is better -> invert),
    # then lower spread (more flexibility is better -> invert), then work density (higher is better),
    # then fewer remaining ops (lower is better -> invert)
    w1 = 0.40  # remaining work
    w2 = 0.20  # soft-min duration (lower is better, so we invert)
    w3 = 0.15  # duration spread (lower is better, so we invert)
    w4 = 0.15  # work density (higher is better)
    w5 = 0.10  # remaining ops (lower is better, so we invert)

    # For duration, spread, and remaining ops, lower is better, so we invert
    score = (w1 * norm_work + 
             w2 * (1.0 - norm_soft_dur) + 
             w3 * (1.0 - norm_spread) + 
             w4 * norm_density + 
             w5 * (1.0 - norm_ops))

    # Deadline guard after computation
    if _deadline_exceeded():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    return int(np.argmax(score))
# EVOLVE-BLOCK-END
