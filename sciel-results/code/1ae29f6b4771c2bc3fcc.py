"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Precompute features for each ready operation
    remaining_works = []
    min_durations = []
    max_durations = []
    remaining_ops_counts = []
    job_ids = []
    machine_option_lists = []

    for op in ready_operations:
        rw = op["remaining_work"]
        remaining_works.append(rw)
        mo = op["machine_options"]
        durations = list(mo.values())
        min_durations.append(min(durations))
        max_durations.append(max(durations))
        remaining_ops_counts.append(op["remaining_ops"])
        job_ids.append(op["job"])
        machine_option_lists.append(list(mo.keys()))

    remaining_works = np.array(remaining_works, dtype=np.float64)
    min_durations = np.array(min_durations, dtype=np.float64)
    max_durations = np.array(max_durations, dtype=np.float64)
    remaining_ops_counts = np.array(remaining_ops_counts, dtype=np.float64)

    # Normalize features to [0, 1] range for fair combination
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    norm_rw = normalize(remaining_works)
    norm_min_d = normalize(min_durations)
    norm_max_d = normalize(max_durations)
    norm_rem_ops = normalize(remaining_ops_counts)

    # Composite heuristic:
    # - Prioritize jobs with more remaining work (to finish long jobs early)
    # - Prioritize operations with longer minimum duration (LPT-like balancing)
    # - Slight penalty for jobs with many remaining ops (to reduce tail length)
    # - Slight preference for operations with smaller max duration (flexibility)
    
    score = (
        0.35 * norm_rw +       # Most remaining work
        0.30 * norm_min_d +    # Longest min duration (LPT)
        0.20 * (1.0 - norm_rem_ops) +  # Fewer remaining ops (closer to completion)
        0.15 * (1.0 - norm_max_d)      # Shorter max duration (more flexible)
    )

    # Tie-breaking: if scores are very close, prefer the one with shorter max duration
    # to reduce potential bottlenecks
    max_score = np.max(score)
    close_mask = score >= max_score - 1e-9
    if np.sum(close_mask) > 1:
        # Among close candidates, pick the one with smallest max duration
        close_indices = np.where(close_mask)[0]
        best_idx = close_indices[np.argmin(max_durations[close_indices])]
    else:
        best_idx = int(np.argmax(score))

    return int(best_idx)
# EVOLVE-BLOCK-END
