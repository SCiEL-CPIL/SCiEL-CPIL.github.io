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


def _soft_min(values, temperature=0.35):
    """
    Smooth surrogate for min() via log-sum-exp (epigraph-friendly soft-min).

    soft_min(v) = - (1/tau) * log( sum_i exp(-tau * v_i) )

    As temperature -> 0 this approaches the true min; a moderate temperature
    gives a smooth objective that also rewards a tight (low-variance) duration
    distribution rather than only the single best machine.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    if v.size == 1:
        return float(v[0])
    tau = max(temperature, 1e-6)
    scaled = -tau * v
    # Numerical stability: shift by max
    m = scaled.max()
    s = np.exp(scaled - m).sum()
    return float(-(m + np.log(s)) / tau)


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.

    Strategy: Combine multiple heuristics to minimize makespan:
    1. Most remaining work (MRW) - prioritize jobs with more work left.
    2. Smooth longest-processing-time (LPT) via a soft-min over machine options,
       so we balance machine loads early while staying smooth in the duration spread.
    3. Remaining operations count - keep pipeline full / complete jobs soon.
    4. Time-aware urgency: operations with less slack relative to current time.
    5. Job-completion proximity: favor operations that finish a job soon.
    6. Machine flexibility: more options = more freedom to avoid bottlenecks.

    We use a weighted combination with a robust tie-breaking mechanism.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Fast path: single ready operation
    if n == 1:
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([max(int(op["remaining_ops"]), 1) for op in ready_operations], dtype=np.float64)

    # Soft-min processing time across machine options (smooth LPT surrogate)
    soft_min_durations = np.array([
        _soft_min(list(op["machine_options"].values()), temperature=0.35) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Average processing time (secondary duration signal)
    avg_durations = np.array([
        float(np.mean(list(op["machine_options"].values()))) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Machine flexibility: number of machine options
    flexibility = np.array([
        len(op["machine_options"]) for op in ready_operations
    ], dtype=np.float64)

    # Normalize features
    def normalize(x):
        rng = x.max() - x.min()
        if rng > 1e-12:
            return (x - x.min()) / rng
        return np.zeros_like(x)

    nw_norm = normalize(remaining_work)
    ro_norm = normalize(remaining_ops)
    smd_norm = normalize(soft_min_durations)  # Higher = longer (LPT)
    ad_norm = normalize(avg_durations)
    flex_norm = normalize(flexibility)

    # Time-aware urgency: how much time has elapsed relative to total work
    if time > 0:
        safe_time = max(time, 1e-6)
        urgency = remaining_work / safe_time
        urgency_norm = normalize(urgency)
    else:
        urgency_norm = nw_norm

    # Job-completion proximity: fewer remaining ops => closer to finishing a job.
    # Use inverse so that "closer to completion" scores higher.
    completion_proximity = 1.0 / remaining_ops
    cp_norm = normalize(completion_proximity)

    # Composite score: we want to maximize this.
    # LPT (via soft-min) helps balance loads early; remaining work keeps critical
    # jobs moving; completion proximity frees resources; flexibility avoids bottlenecks.
    w_rw = 0.35   # weight for remaining work
    w_smd = 0.30  # weight for soft-min duration (smooth LPT)
    w_ro = 0.10   # weight for remaining ops
    w_urg = 0.10  # weight for time-aware urgency
    w_cp = 0.10   # weight for job-completion proximity
    w_flex = 0.05 # weight for machine flexibility

    score = (w_rw * nw_norm +
             w_smd * smd_norm +
             w_ro * ro_norm +
             w_urg * urgency_norm +
             w_cp * cp_norm +
             w_flex * flex_norm)

    # Safety: if something goes wrong, fall back to max remaining work
    if not _elapsed_ok():
        return int(np.argmax(remaining_work))

    # Find the best index
    idx = int(np.argmax(score))

    # Robust tie-breaking:
    # If scores are very close, prefer the operation with the longest soft-min
    # duration (LPT consistency), then more flexibility, then more remaining work.
    max_score = score.max()
    close = np.where(score > max_score - 1e-6)[0]
    if len(close) > 1:
        # Among close candidates, pick the one with the longest soft-min duration,
        # breaking further ties by flexibility then remaining work.
        close_smd = soft_min_durations[close]
        close_flex = flexibility[close]
        close_rw = remaining_work[close]

        # Composite tie-break key (higher is better)
        tie_key = (close_smd * 1.0 +
                   close_flex * 1e-3 +
                   close_rw * 1e-6)
        best_local = int(np.argmax(tie_key))
        idx = int(close[best_local])

    return int(idx)
# EVOLVE-BLOCK-END
