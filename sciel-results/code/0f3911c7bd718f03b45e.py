"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


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

    # STRUCTURAL CHANGE: Simulation-based lookahead scoring.
    # For each candidate (op, machine) pair, simulate the immediate scheduling impact:
    # 1. The machine will be busy for `duration` units.
    # 2. Other ready operations that also want this machine are delayed by `duration`.
    # 3. Score = urgency of this op - estimated delay imposed on others.
    # This directly models machine contention rather than using static proxies.

    # Pre-compute features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ])
    flexibility = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=float)

    # Normalize urgency
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    normalized_urgency = remaining_works / (max_work + 1e-9)

    # Normalize flexibility
    max_flex = np.max(flexibility) if np.max(flexibility) > 0 else 1.0
    normalized_flex = flexibility / (max_flex + 1e-9)

    # For each ready op, find the best (op, machine) combination via lookahead
    best_score = -np.inf
    best_idx = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            continue

        # For each eligible machine, estimate the lookahead cost
        for m, dur in mo.items():
            # Estimate delay imposed on other ready ops that also need machine m
            # Sum of durations of other ops that can use machine m
            # (they will be blocked while this op runs)
            delay_imposed = 0.0
            for j, other_op in enumerate(ready_operations):
                if j == i:
                    continue
                other_mo = other_op["machine_options"]
                if m in other_mo:
                    # Other op could be delayed by up to dur if it also wants m
                    # Weight by how urgent that other op is
                    delay_imposed += normalized_urgency[j] * dur

            # Score: high urgency + flexibility - delay imposed on others
            # The delay term is scaled down since it's a soft penalty
            delay_penalty = 0.3 * delay_imposed
            score = (1.5 * normalized_urgency[i] +
                     0.3 * normalized_flex[i] -
                     delay_penalty)

            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = m

    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
