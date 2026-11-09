"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard: 150s wall-clock limit, keep a safety margin.
_DEADLINE = _time.time() + 145.0


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    Returns an integer index into ready_operations.
    """
    # Safety: if we're close to the deadline, return a quick heuristic.
    if _time.time() > _DEADLINE - 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    job_ids = np.array([op["job"] for op in ready_operations], dtype=np.int64)

    # For each op, compute the minimum duration across eligible machines
    min_durs = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=np.float64)
    max_durs = np.array([max(op["machine_options"].values()) for op in ready_operations], dtype=np.float64)
    num_machines = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)

    # --- Heuristic scoring ---
    # We want to minimize makespan. Key ideas:
    # 1. Jobs with more remaining work should be prioritized (critical path).
    # 2. Operations with shorter processing times reduce idle time.
    # 3. Jobs with fewer remaining operations are closer to completion (tail).
    # 4. Balance load: avoid concentrating work on few jobs.

    # Normalize features to [0, 1] range to make weights comparable
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    nw = normalize(remaining_works)
    no = normalize(remaining_ops)
    nd_min = normalize(min_durs)
    nd_max = normalize(max_durs)
    nm = normalize(num_machines)

    # Score: higher is better (we'll pick the max score)
    # Weighted combination:
    # - High remaining work -> prioritize (avoids starvation)
    # - Short min duration -> prioritize (reduces makespan)
    # - Few remaining ops -> prioritize (completes jobs, frees resources)
    # - Many machine options -> slightly prefer (flexibility)

    score = (
        0.35 * nw          # remaining work
      + 0.30 * (1.0 - nd_min)  # short processing time
      + 0.20 * (1.0 - no)      # few remaining ops (closer to done)
      + 0.10 * nm              # flexibility
      + 0.05 * (1.0 - nd_max)  # low max duration (less risk)
    )

    idx = int(np.argmax(score))

    # Tie-breaking: among top candidates, prefer the one with the shortest
    # actual minimum duration to reduce immediate makespan contribution.
    top_candidates = np.where(score >= score[idx] - 1e-9)[0]
    if len(top_candidates) > 1:
        # Among ties, pick the one with shortest min duration, then most remaining work
        sub_min_dur = min_durs[top_candidates]
        sub_rem_work = remaining_works[top_candidates]
        best_sub = np.lexsort((-sub_rem_work, sub_min_dur))
        idx = int(top_candidates[best_sub])

    return idx
# EVOLVE-BLOCK-END
