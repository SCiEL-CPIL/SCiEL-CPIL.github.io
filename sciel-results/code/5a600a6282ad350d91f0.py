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
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
            machine_counts[i] = len(mo)
        else:
            min_durations[i] = 0.0
            machine_counts[i] = 1.0

    # Helper function for min-max normalization
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        else:
            return np.ones_like(arr) / len(arr)

    # Feature 1: WSPT Ratio (Remaining Work / Min Duration)
    # High WSPT indicates high urgency (lots of work compressed into short time slots)
    eps = 1e-9
    wspt_raw = remaining_works / (min_durations + eps)
    wspt_norm = normalize(wspt_raw)

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

    # Feature 5: SPT Bonus (Inverse Min Duration)
    # Prefer shorter operations to reduce immediate blocking time
    spt_raw = 1.0 / (min_durations + 1.0)
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
    w_wspt = 0.30      # Primary urgency
    w_rw = 0.20        # Critical path load
    w_ro = 0.15        # Keep-in-motion / Chain length
    w_op = 0.10        # Sequence criticality
    w_spt = 0.10       # Short processing time preference
    w_constraint = 0.15 # Bottleneck / Flexibility constraint

    scores = (
        w_wspt * wspt_norm
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
