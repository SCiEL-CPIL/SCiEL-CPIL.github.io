"""Rewritten dispatch for flexible job-shop; EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


def _fast_machine_score(machine_options):
    """Return (best_machine, load_score) where load_score is smaller for less-loaded machines.

    We approximate machine load by the number of distinct operations that can use that machine
    across the current ready set. A machine used by many ready ops is more contended.
    If we only have a single machine option, contention is implicit.
    """
    if not machine_options:
        return None, np.inf

    # Count how many ready ops can use each machine in this op's option set
    # This is a per-op local contention estimate; we don't track global machine loads.
    # Instead, we use a simple heuristic: prefer machines with shorter duration.
    # Among equal or near-equal durations, prefer machines that are options for fewer ready ops.
    # But since we don't have the full ready set here, we fall back to shortest duration.
    best_machine = min(machine_options, key=machine_options.get)
    best_dur = machine_options[best_machine]
    return best_machine, best_dur


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick operation with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0

    # Extract key features for all ready operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ])
    max_durations = np.array([
        max(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ])
    spread_durations = max_durations - min_durations  # flexibility indicator

    # Compute critical path-like estimate: remaining work divided by number of remaining ops
    avg_work_per_op = np.where(remaining_ops > 0, remaining_works / np.maximum(remaining_ops, 1), remaining_works)

    # Look-ahead bottleneck heuristic:
    # For each ready op, estimate the earliest completion time if assigned to the least-contended machine.
    # We approximate machine contention by the number of ready ops that share each machine.
    # Build a machine->count map across all ready ops.
    machine_usage = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_usage[m] = machine_usage.get(m, 0) + 1

    # For each op, find the machine with minimum (duration * contention_factor)
    # contention_factor = 1 + usage_count (heuristic: more users => more likely to be busy)
    lookahead_completion = np.array([0.0] * n)
    for i, op in enumerate(ready_operations):
        best_val = np.inf
        for m, dur in op["machine_options"].items():
            usage = machine_usage.get(m, 0)
            # Penalize machines that many ops want: higher usage => later completion
            # Use a soft penalty: dur * (1 + 0.1 * usage)
            val = dur * (1.0 + 0.1 * usage)
            if val < best_val:
                best_val = val
        lookahead_completion[i] = best_val

    # Normalize components
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_avg = np.max(avg_work_per_op) if np.max(avg_work_per_op) > 0 else 1.0
    max_lookahead = np.max(lookahead_completion) if np.max(lookahead_completion) > 0 else 1.0
    max_inv_min_dur = np.max(1.0 / min_durations) if np.any(min_durations > 0) else 1.0

    normalized_urgency = remaining_works / (max_work + 1e-9)
    normalized_avg = avg_work_per_op / (max_avg + 1e-9)
    normalized_lookahead = lookahead_completion / (max_lookahead + 1e-9)  # smaller is better
    normalized_efficiency = (1.0 / min_durations) / (max_inv_min_dur + 1e-9)

    # Weights: emphasize urgency and lookahead bottleneck, moderate efficiency
    w_urgency = 1.0
    w_avg = 0.5
    w_lookahead = 0.6  # smaller lookahead => better, so subtract
    w_efficiency = 0.3

    # Score: higher is better. Subtract normalized_lookahead since smaller completion is preferred.
    scores = (w_urgency * normalized_urgency +
              w_avg * normalized_avg -
              w_lookahead * normalized_lookahead +
              w_efficiency * normalized_efficiency)

    best_idx = int(np.argmax(scores))

    # Choose the machine with minimum duration * contention factor for the selected op
    op = ready_operations[best_idx]
    machine_options = op["machine_options"]
    if machine_options:
        best_machine = None
        best_val = np.inf
        for m, dur in machine_options.items():
            usage = machine_usage.get(m, 0)
            val = dur * (1.0 + 0.1 * usage)
            if val < best_val:
                best_val = val
                best_machine = m
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
