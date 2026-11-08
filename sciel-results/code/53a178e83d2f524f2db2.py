"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Returns:
        int: index of the ready operation to start
        or (int, machine): index and chosen machine if multiple options
    """
    if not ready_operations:
        return 0  # Should not happen per spec, but safe fallback
    
    # Deadline guard: if we're running low on time, use a fast heuristic
    # The evaluator kills after 150s, so we check elapsed time
    # Since we don't have a global start time, we use a simple fast path
    # for very large ready_operations lists
    
    n = len(ready_operations)
    
    # Fast path: if many operations, use a simpler heuristic to avoid timeout
    if n > 100:
        # Use shortest processing time (min duration across machines)
        scores = []
        for op in ready_operations:
            min_dur = min(op["machine_options"].values())
            # Combine with remaining work as tiebreaker
            scores.append((min_dur, -op["remaining_work"]))
        return int(np.argmin(scores))
    
    # Compute scores for each ready operation
    # Heuristic: minimize a combination of:
    # 1. Shortest processing time (SPT) - prefer operations that finish quickly
    # 2. Most remaining work - prefer operations from jobs with more work left
    # 3. Fewest remaining operations - prefer operations that are closer to completion
    #    (actually, we want to unlock more, so more remaining ops might be better
    #     for parallelism, but completing a job frees up resources)
    
    # Let's use a weighted score:
    # score = w1 * normalized_min_duration - w2 * normalized_remaining_work + w3 * normalized_remaining_ops
    
    # First, gather features
    min_durs = []
    remaining_works = []
    remaining_ops = []
    
    for op in ready_operations:
        min_durs.append(min(op["machine_options"].values()))
        remaining_works.append(op["remaining_work"])
        remaining_ops.append(op["remaining_ops"])
    
    # Normalize to [0, 1] range
    def normalize(arr):
        arr = np.array(arr, dtype=float)
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_durs = normalize(min_durs)
    norm_works = normalize(remaining_works)
    norm_ops = normalize(remaining_ops)
    
    # Weighted score: lower is better
    # We want to minimize duration, maximize remaining work (to unlock more),
    # and the effect of remaining ops is ambiguous - let's use a small weight
    # Actually, for makespan minimization, SPT is known to be good.
    # Let's primarily use SPT with a tiebreaker toward more remaining work.
    
    # score = norm_dur - 0.1 * norm_work + 0.05 * norm_ops
    # Lower score is better
    
    scores = 1.0 * norm_durs - 0.1 * norm_works + 0.05 * norm_ops
    
    # Find the operation with the lowest score
    best_idx = int(np.argmin(scores))
    
    # Optionally, choose the best machine for the selected operation
    # The task says "optionally, which eligible machine runs it"
    # Returning (index, machine) might be useful if the simulator supports it
    # But the current program just returns an index, so let's stick with that
    # unless we have evidence that machine selection matters
    
    # For now, return just the index to match the original interface
    return best_idx
# EVOLVE-BLOCK-END
