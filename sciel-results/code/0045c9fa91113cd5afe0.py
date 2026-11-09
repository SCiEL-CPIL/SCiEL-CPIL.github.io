"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = _time.time()
    return _time.time() - _START


def _critical_path_length(op):
    """
    Estimate the critical path length for the job containing this operation.
    Uses remaining_work and remaining_ops to estimate how much work is left
    on the critical path. Longer critical path = higher priority.
    """
    rem_work = op.get("remaining_work", 0)
    rem_ops = op.get("remaining_ops", 0)
    mo = op.get("machine_options", {})

    if mo:
        min_proc = min(mo.values())
    else:
        min_proc = 0

    # Critical path estimate: remaining work plus a penalty for remaining ops
    # (each remaining op adds setup/transition overhead)
    crit_path = rem_work + 0.1 * max(rem_ops, 0)
    return crit_path


def _soft_min_machine(op, temperature=0.5):
    """
    Choose a machine using a soft-min (softmax over negative durations) to
    spread load when durations are close, while still preferring shorter ones.
    Returns the machine index.
    """
    mo = op["machine_options"]
    if not mo:
        return 0

    machines = list(mo.keys())
    durations = np.array([mo[m] for m in machines], dtype=np.float64)

    if len(machines) == 1:
        return machines[0]

    # Soft-min: use softmax with negative durations (lower duration = higher weight)
    # Temperature controls how "soft" the selection is
    neg_durs = -durations
    max_neg = np.max(neg_durs)
    exp_vals = np.exp((neg_durs - max_neg) / temperature)
    probs = exp_vals / np.sum(exp_vals)

    # Pick the machine with highest probability (deterministic: argmax of probs)
    best_idx = int(np.argmax(probs))
    return machines[best_idx]


def _score_operation(op, time):
    """
    Composite score: higher is better (we pick max).
    Balances:
      - critical_path_length (urgency: jobs with longer remaining critical path start early)
      - remaining_ops (fewer remaining ops = closer to completion, reduce WIP)
      - soft-min processing time (SPT-like but smoother)
    """
    rem_work = op.get("remaining_work", 0)
    rem_ops = op.get("remaining_ops", 0)
    mo = op.get("machine_options", {})

    # Critical path length
    crit_path = _critical_path_length(op)

    # Soft-min processing time: use the minimum duration as a proxy,
    # but blend with the mean to avoid over-penalizing ops that only
    # have one machine option
    if mo:
        durs = list(mo.values())
        min_proc = min(durs)
        mean_proc = sum(durs) / len(durs)
        # Blend: 70% min, 30% mean to soften the impact
        smooth_proc = 0.7 * min_proc + 0.3 * mean_proc
    else:
        smooth_proc = 0

    # Score components
    score = 1.0 * crit_path
    score += 0.3 * (rem_ops if rem_ops > 0 else 1)
    score -= 0.5 * smooth_proc

    return score


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if we've used >110s, return immediately with a fast heuristic
    if _elapsed() > 110.0:
        # Fast fallback: just pick max remaining_work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Compute composite scores
    scores = [_score_operation(op, time) for op in ready_operations]
    best_idx = int(np.argmax(scores))

    # For the chosen operation, pick the best machine using soft-min
    chosen_op = ready_operations[best_idx]
    machine = _soft_min_machine(chosen_op, temperature=0.5)

    return (best_idx, machine)
# EVOLVE-BLOCK-END
