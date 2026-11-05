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

    Strategy: machine-load balancing. For each ready operation, we look at its
    machine_options. We prefer operations that target the most congested machines
    but have short durations, to prevent single-machine bottlenecks.

    We compute a score for each ready operation:
      - For each operation, pick the best (shortest) machine option.
      - Score = (machine_congestion * 0.3) + (duration * 0.7) - (remaining_work * 0.2)
      where machine_congestion is estimated from the current time and how much
      work other ready ops target the same machine.

    Since we don't have global machine load info, we approximate congestion by
    counting how many other ready operations also target the same machine.

    We use a weighted combination:
      - Shortest duration on the chosen machine (prefer shorter)
      - Fewer competing ready ops on the same machine (prefer less contention)
      - More remaining work for the job (prefer to keep jobs progressing)

    We pick the operation with the highest composite score (i.e., best to start now).
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Build machine -> list of (op_index, duration) for quick contention counting
    machine_contention = {}
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            continue
        # Use the shortest-duration machine as the "target"
        best_machine = min(mo.keys(), key=lambda m: mo[m])
        best_dur = mo[best_machine]
        if best_machine not in machine_contention:
            machine_contention[best_machine] = []
        machine_contention[best_machine].append((i, best_dur))

    # Compute scores
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if not mo:
            scores[i] = -1e9
            continue

        # Pick the best machine (shortest duration)
        best_machine = min(mo.keys(), key=lambda m: mo[m])
        best_dur = mo[best_machine]

        # Contention: number of other ops targeting the same machine
        contention = len(machine_contention.get(best_machine, [])) - 1  # exclude self

        # Remaining work for the job
        rem_work = op.get("remaining_work", 0)

        # Composite score: we want to START this op if it reduces bottleneck risk.
        # Higher score = better to start now.
        # - Lower duration is better (finish faster, free the machine)
        # - Lower contention is better (less queueing on that machine)
        # - Higher remaining work means the job has more to do; starting now
        #   helps keep the job progressing (LRPT-like component)
        #
        # We normalize-ish by using raw values since we don't know scales.
        # Use a weighted sum where lower duration and lower contention are good.
        # Invert: score = -alpha*duration - beta*contention + gamma*rem_work

        alpha = 1.0
        beta = 0.5
        gamma = 0.3

        scores[i] = -alpha * best_dur - beta * contention + gamma * rem_work

    # Safety: if all scores are equal or something went wrong, fall back to argmax remaining_work
    if np.allclose(scores, scores[0]):
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
