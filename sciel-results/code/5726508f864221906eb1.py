"""Rewritten program for open-shop scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE_SECONDS = 145.0
_start_time = _time.monotonic()


def _elapsed():
    return _time.monotonic() - _start_time


def dispatch(ready_operations, time):
    """Pick which ready operation to start next.

    Strategy: Longest Processing Time (LPT) with load balancing.
    We prioritize operations with longer processing times on their best machine,
    as these are likely to be critical for the makespan. We also penalize
    operations that target highly congested machines to prevent bottlenecks.
    
    Score = duration - penalty * contention + bonus * remaining_work
    We pick the operation with the highest score.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # 1. Determine the "best" machine for each operation (shortest duration)
    # and the duration on that machine.
    best_machines = [None] * n
    best_durations = [0.0] * n
    remaining_works = [0.0] * n

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        rem_work = op.get("remaining_work", 0.0)
        remaining_works[i] = rem_work
        
        if not mo:
            best_machines[i] = -1
            best_durations[i] = 0.0
            continue

        # Find machine with minimum duration
        min_dur = float('inf')
        best_m = -1
        for m, dur in mo.items():
            if dur < min_dur:
                min_dur = dur
                best_m = m
        
        best_machines[i] = best_m
        best_durations[i] = min_dur

    # 2. Calculate contention (load) for each machine
    # Load is the sum of durations of all ready ops targeting that machine.
    machine_load = {}
    for i in range(n):
        m = best_machines[i]
        if m == -1:
            continue
        if m not in machine_load:
            machine_load[m] = 0.0
        machine_load[m] += best_durations[i]

    # 3. Compute scores
    scores = np.zeros(n)
    for i in range(n):
        m = best_machines[i]
        if m == -1:
            scores[i] = -1e9
            continue

        dur = best_durations[i]
        load = machine_load.get(m, 0.0)
        # Contention is the load of OTHER ops on this machine
        contention = load - dur
        
        rem_work = remaining_works[i]

        # LPT: Favor longer durations.
        # Load Balance: Penalize high contention.
        # MWR: Small bonus for jobs with more work left.
        
        # Weights:
        # alpha = 1.0 (LPT weight)
        # beta = 0.2 (Load balance penalty)
        # gamma = 0.1 (MWR bonus)
        
        alpha = 1.0
        beta = 0.2
        gamma = 0.1

        scores[i] = alpha * dur - beta * contention + gamma * rem_work

    # Safety: if all scores are equal or something went wrong, fall back to argmax remaining_work
    if np.allclose(scores, scores[0]):
        return int(np.argmax(remaining_works))

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
