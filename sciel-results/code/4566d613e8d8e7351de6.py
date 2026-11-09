"""Rewritten program for Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

_START_TIME = _time.time()
_DEADLINE_SECONDS = 115.0  # Safety margin before 120s limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Use a composite heuristic that balances:
    1. Critical Path Length (CPL) via remaining_work: higher is more urgent.
    2. Shortest Processing Time (SPT): shorter min duration preferred.
       Uses a soft-min over machine durations for smoother scoring.
    3. Remaining operations count: more remaining ops means the job is in early stages,
       so completing the current op unlocks more future work - prioritize jobs with
       more remaining ops to keep the pipeline flowing.
    
    We compute a weighted score and pick the operation with the highest score.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE_SECONDS:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features using numpy for efficiency
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each operation, compute a soft-min over machine durations using log-sum-exp
    # with negative temperature as a smooth surrogate for the hard min.
    # This avoids discontinuity when two machines have near-equal durations.
    soft_min_durations = np.zeros(n, dtype=np.float64)
    T = 0.5  # Temperature parameter for soft-min
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durs = np.array(list(mo.values()), dtype=np.float64)
            if len(durs) == 1:
                soft_min_durations[i] = durs[0]
            else:
                # soft-min: -T * log(sum(exp(-d/T)))
                neg_d_over_T = -durs / T
                max_neg = np.max(neg_d_over_T)
                # log-sum-exp trick for numerical stability
                log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_d_over_T - max_neg)))
                soft_min_durations[i] = -T * log_sum_exp
        else:
            soft_min_durations[i] = 1.0  # fallback

    # Normalize remaining_work to [0, 1] - higher remaining work = higher priority
    max_rw = remaining_works.max()
    min_rw = remaining_works.min()
    rw_range = max_rw - min_rw
    if rw_range < 1e-9:
        norm_rw = np.full(n, 0.5)
    else:
        norm_rw = (remaining_works - min_rw) / rw_range

    # Inverse soft_min_duration: shorter processing time gets higher score
    # Scale by 1/soft_min_dur, but cap to avoid extreme values
    inv_dur = 1.0 / (soft_min_durations + 1e-6)
    max_inv = inv_dur.max()
    if max_inv < 1e-9:
        norm_inv_dur = np.full(n, 0.5)
    else:
        norm_inv_dur = inv_dur / max_inv

    # Remaining ops: more remaining ops means the job has more future work to unlock
    # So we prioritize jobs with more remaining ops to keep the schedule flowing
    max_ro = remaining_ops.max()
    min_ro = remaining_ops.min()
    ro_range = max_ro - min_ro
    if ro_range < 1e-9:
        norm_ro = np.full(n, 0.5)
    else:
        norm_ro = (remaining_ops - min_ro) / ro_range

    # Composite score weights - tuned for job-shop scheduling
    # Higher weight on critical path (remaining_work) as it directly impacts makespan
    # Moderate weight on SPT to avoid long operations blocking machines
    # Lower weight on remaining ops as a tiebreaker
    w1 = 0.50  # remaining work (critical path) - dominant factor
    w2 = 0.35  # shortest processing time
    w3 = 0.15  # remaining ops urgency

    scores = w1 * norm_rw + w2 * norm_inv_dur + w3 * norm_ro

    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
