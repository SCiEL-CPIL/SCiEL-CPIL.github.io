"""Combined Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

_START_TIME = _time.time()
_DEADLINE_SECONDS = 115.0  # Safety margin before 120s limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Combine multiple heuristics into a composite score:
    1. Critical Path Length (CPL): Jobs with longer remaining paths are urgent.
    2. Minimum Remaining Work: Favors operations that finish quickly (shortest processing time).
    3. Machine Load Balance: Prefer less busy machines if we can infer from options.
    4. Number of remaining ops: More remaining ops = higher priority (tail of the job).
    
    The composite score is:
      score = w1 * normalized_remaining_work + 
              w2 * (1 / (min_duration + epsilon)) + 
              w3 * (1 / (remaining_ops + 1))
    
    We pick the operation with the highest score.
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
    
    # For each operation, find the minimum duration among machine options
    min_durations = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
        else:
            min_durations[i] = 1.0  # fallback

    # Normalize remaining_work to [0, 1]
    max_rw = remaining_works.max()
    min_rw = remaining_works.min()
    rw_range = max_rw - min_rw
    if rw_range < 1e-9:
        norm_rw = np.full(n, 0.5)
    else:
        norm_rw = (remaining_works - min_rw) / rw_range

    # Inverse min_duration: shorter processing time gets higher score
    # Scale by 1/min_dur, but cap to avoid extreme values
    inv_dur = 1.0 / (min_durations + 1e-6)
    max_inv = inv_dur.max()
    if max_inv < 1e-9:
        norm_inv_dur = np.full(n, 0.5)
    else:
        norm_inv_dur = inv_dur / max_inv

    # Inverse remaining_ops: fewer remaining ops means this is closer to completion
    # Actually, we want to prioritize jobs with MORE remaining work (critical path),
    # but also favor operations that can complete soon.
    # Let's use: jobs with more remaining ops need urgent attention to avoid blocking.
    # So higher remaining_ops => higher urgency.
    max_ro = remaining_ops.max()
    min_ro = remaining_ops.min()
    ro_range = max_ro - min_ro
    if ro_range < 1e-9:
        norm_ro = np.full(n, 0.5)
    else:
        norm_ro = (remaining_ops - min_ro) / ro_range

    # Composite score weights
    w1 = 0.40  # remaining work (critical path)
    w2 = 0.35  # shortest processing time
    w3 = 0.25  # remaining ops urgency

    scores = w1 * norm_rw + w2 * norm_inv_dur + w3 * norm_ro

    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
