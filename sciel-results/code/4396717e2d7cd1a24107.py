"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock.
# Keep a safety margin; if we exceed it, fall back to a fast heuristic.
_DEADLINE = 145.0
_START_TIME = _time.monotonic()


def _check_deadline():
    return _time.monotonic() - _START_TIME > _DEADLINE


def _fast_dispatch(ready_operations, time):
    """Fast fallback: shortest processing time first (SPT) on min machine option."""
    if not ready_operations:
        return 0
    durations = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durations.append(min(mo.values()))
        else:
            durations.append(0.0)
    return int(np.argmin(durations))


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    if _check_deadline():
        return _fast_dispatch(ready_operations, time)

    n = len(ready_operations)

    # For each ready operation, compute a composite score.
    # We want to pick the operation that best reduces makespan.
    # Heuristics combined:
    #  1) Shortest processing time (SPT) on its best machine option -> reduces
    #     critical path length.
    #  2) Job with most remaining work (remaining_work) -> prioritize long jobs
    #     to avoid them being stuck at the end.
    #  3) Machine load balance: prefer machines that are less loaded. We don't
    #     have direct machine load, but we can approximate by preferring the
    #     machine option with the shortest duration (which also tends to be
    #     less contended in a well-scheduled shop).
    #  4) remaining_ops: jobs with fewer remaining ops are closer to completion;
    #     finishing them can free resources.

    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_dur = min(mo.values())
            max_dur = max(mo.values())
            # Use min duration as the effective processing time on best machine.
            proc_time = min_dur
        else:
            proc_time = 0.0
            max_dur = 0.0

        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)

        # Normalize-ish components (we don't know global scales, so use raw).
        # We want: high remaining_work (long job) -> good, but also short proc_time.
        # Combine: score = remaining_work - alpha * proc_time
        # Higher score = better.
        # But we also want to avoid starving: if a job has very few remaining ops,
        # completing it quickly is good.

        # A common effective heuristic: 
        #   score = remaining_work / (1 + proc_time)
        # This rewards long jobs but penalizes long processing times.

        score = remaining_work / (1.0 + proc_time)

        # Add a small bonus for jobs with fewer remaining ops (closer to done),
        # to help clear the shop.
        if remaining_ops > 0:
            score += 1.0 / remaining_ops

        scores[i] = score

    # Pick the operation with the highest composite score.
    # In case of ties, prefer the one with shortest processing time.
    best_idx = int(np.argmax(scores))

    # Tie-breaking: if multiple have same score, pick shortest proc time.
    max_score = scores[best_idx]
    tied = [i for i in range(n) if abs(scores[i] - max_score) < 1e-9]
    if len(tied) > 1:
        tied_durs = []
        for i in tied:
            mo = ready_operations[i]["machine_options"]
            tied_durs.append(min(mo.values()) if mo else 0.0)
        best_idx = tied[int(np.argmin(tied_durs))]

    return best_idx
# EVOLVE-BLOCK-END
