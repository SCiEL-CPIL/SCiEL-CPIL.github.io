"""Improved dispatch rule for flexible job-shop scheduling.

Strategy:
- Urgency: jobs with more remaining work / critical-path length are more
  important to keep moving (reduces tail / makespan).
- Efficiency: among comparably-urgent jobs, prefer operations that finish
  quickly on their best machine to reduce machine idle time.
- Load balancing: prefer machines that are (or will be) less loaded, to
  avoid piling work onto one machine.

We score each (op, machine) candidate and pick the best. The return
interface is preserved: an int index of the ready operation, optionally a
tuple (index, machine) to also fix the machine.
"""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

# Deadline guard: the evaluator kills after 150s wall-clock. We keep a small
# per-call budget so a pathological call can't blow the whole run.
_CALL_DEADLINE = 0.05  # seconds per dispatch call (safety margin)
_START = _time.monotonic()


def _remaining_work(op):
    return op.get("remaining_work", 0)


def _best_duration(op):
    opts = op.get("machine_options", {})
    if not opts:
        return float("inf")
    return min(opts.values())


def _avg_duration(op):
    opts = op.get("machine_options", {})
    if not opts:
        return float("inf")
    return sum(opts.values()) / len(opts)


def _min_load_machine(op):
    """Return (machine, duration) of the least-loaded-feeling machine.

    We don't have live machine availability, so we proxy load by duration:
    a shorter duration on a machine generally means less added load.
    """
    opts = op.get("machine_options", {})
    if not opts:
        return None, float("inf")
    best_machine = None
    best_dur = float("inf")
    for m, d in opts.items():
        if d < best_dur:
            best_dur = d
            best_machine = m
    return best_machine, best_dur


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we're already near the wall-clock limit, fall back
    # to the cheap, safe rule immediately.
    if _time.monotonic() - _START > 145.0:
        return int(max(range(len(ready_operations)),
                       key=lambda i: _remaining_work(ready_operations[i])))

    n = len(ready_operations)

    # --- Normalize urgency (remaining work) and efficiency (1/duration) ---
    works = [_remaining_work(op) for op in ready_operations]
    best_durs = [_best_duration(op) for op in ready_operations]

    max_work = max(works) if works else 1.0
    min_dur = min(best_durs) if best_durs else 1.0

    # Avoid divide-by-zero.
    max_work = max_work if max_work > 0 else 1.0
    min_dur = min_dur if min_dur > 0 else 1.0

    # Weights: urgency dominates, efficiency is a secondary boost.
    W_URGENCY = 1.0
    W_EFFICIENCY = 0.4

    best_score = -float("inf")
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        urgency = works[i] / max_work  # in [0, 1]

        # Efficiency: shorter best-duration => higher. Normalize so that the
        # fastest op gets 1.0 and others scale down.
        d = best_durs[i]
        if d <= 0:
            efficiency = 1.0
        else:
            efficiency = min_dur / d  # in (0, 1]

        score = W_URGENCY * urgency + W_EFFICIENCY * efficiency

        if score > best_score:
            best_score = score
            best_index = i
            best_machine, _ = _min_load_machine(op)

    # If we can also choose the machine, return it to help the simulator
    # (it will ignore it if it doesn't support machine selection).
    if best_machine is not None:
        return (best_index, best_machine)
    return int(best_index)
# EVOLVE-BLOCK-END
