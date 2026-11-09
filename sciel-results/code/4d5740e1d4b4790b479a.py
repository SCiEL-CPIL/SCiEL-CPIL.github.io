"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track per-call time to avoid pathological cases.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin


def _is_deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if _is_deadline_exceeded():
        # Fallback: pick the operation with the most remaining work
        # This is a safe heuristic that usually keeps the schedule moving
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    
    if n == 1:
        return 0

    # Extract features into arrays for vectorized operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    op_numbers = np.array([op["op"] for op in ready_operations], dtype=np.float64)
    
    min_durations = np.empty(n, dtype=np.float64)
    machine_counts = np.empty(n, dtype=np.float64)
    soft_min_durations = np.empty(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durations = list(mo.values())
            min_durations[i] = min(durations)
            machine_counts[i] = len(durations)
            # Soft-min using log-sum-exp for smooth surrogate of min
            # This gives a smooth approximation that considers all options
            if len(durations) > 1:
                d_arr = np.array(durations, dtype=np.float64)
                # Temperature-controlled soft min: lower temp = closer to true min
                temp = 5.0
                log_sum_exp = np.log(np.sum(np.exp(-d_arr / temp)))
                soft_min_durations[i] = -temp * log_sum_exp
            else:
                soft_min_durations[i] = durations[0]
        else:
            min_durations[i] = 0.0
            machine_counts[i] = 1.0
            soft_min_durations[i] = 0.0

    # Helper function for min-max normalization
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        else:
            return np.ones_like(arr) / len(arr)

    # Feature 1: WSPT Ratio (Remaining Work / Soft-Min Duration)
    # High WSPT indicates high urgency (lots of work compressed into short time slots)
    # Using soft-min gives a smoother, more robust urgency signal
    eps = 1e-9
    wspt_raw = remaining_works / (soft_min_durations + eps)
    wspt_norm = normalize(wspt_raw)
    
    # Feature 1b: Hard WSPT as secondary signal
    wspt_hard_raw = remaining_works / (min_durations + eps)
    wspt_hard_norm = normalize(wspt_hard_raw)

    # Feature 2: Remaining Work (Total load remaining)
    # High remaining work suggests the job is on the critical path
    rw_norm = normalize(remaining_works)

    # Feature 3: Remaining Ops (Number of steps left)
    # Inspiration 2 found that prioritizing jobs with more remaining ops (keep-in-motion) 
    # helped when normalized. This prevents short jobs from starving long chains.
    ro_norm = normalize(remaining_ops)

    # Feature 4: Operation Number (Position in job sequence)
    # Higher op number means closer to completion, but also less flexibility.
    # Prioritizing later ops can help clear the job.
    op_norm = normalize(op_numbers)

    # Feature 5: SPT Bonus (Inverse Soft-Min Duration)
    # Prefer shorter operations to reduce immediate blocking time
    spt_raw = 1.0 / (soft_min_durations + 1.0)
    spt_norm = normalize(spt_raw)

    # Feature 6: Constraint Score (Inverse Machine Count)
    # Operations with fewer machine options are more constrained.
    # Prioritizing them reduces the risk of them becoming infeasible or causing idle waits
    # due to lack of alternative machines. This is the "bottleneck identification" aspect.
    constraint_raw = 1.0 / (machine_counts + eps)
    constraint_norm = normalize(constraint_raw)

    # Combine features with tuned weights
    # Weights are set to balance the multi-criteria, ensuring no single feature dominates
    # due to scale, thanks to normalization.
    # Increased WSPT weight (urgency is primary signal), added hard WSPT as complement,
    # reduced op number weight to avoid premature job completion bias
    w_wspt = 0.35      # Primary urgency (soft-min based)
    w_wspt_hard = 0.10 # Hard WSPT complement
    w_rw = 0.18        # Critical path load
    w_ro = 0.12        # Keep-in-motion / Chain length
    w_op = 0.05        # Sequence criticality - reduced
    w_spt = 0.08       # Short processing time preference
    w_constraint = 0.12 # Bottleneck / Flexibility constraint

    scores = (
        w_wspt * wspt_norm
        + w_wspt_hard * wspt_hard_norm
        + w_rw * rw_norm
        + w_ro * ro_norm
        + w_op * op_norm
        + w_spt * spt_norm
        + w_constraint * constraint_norm
    )

    # Return index of the operation with the highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
