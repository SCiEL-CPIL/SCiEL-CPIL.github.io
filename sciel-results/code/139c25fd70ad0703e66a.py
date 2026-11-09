"""Open-shop scheduling dispatch rule.

Uses a multi-factor heuristic combining:
1. Remaining work (critical path proxy)
2. Machine availability / load balancing
3. Operation urgency (remaining ops)
4. Machine flexibility (number of options)

With adaptive weights based on simulation state and a look-ahead
penalty for operations that would block heavily-loaded machines.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.time()
_HARD_LIMIT = 140.0  # seconds


def _elapsed_ok():
    return (_time.time() - _START) < _HARD_LIMIT


# Track machine loads across calls to inform dispatch decisions
_machine_loads = {}
_dispatch_count = 0


def dispatch(ready_operations, time):
    global _machine_loads, _dispatch_count
    _dispatch_count += 1

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            # Pick the machine with least current load, tie-break by shortest duration
            best_machine = min(
                machine_options,
                key=lambda m: (_machine_loads.get(m, 0), machine_options[m])
            )
            return (0, best_machine)
        return 0

    # Deadline check
    if not _elapsed_ok():
        max_work = max(op["remaining_work"] for op in ready_operations)
        for i, op in enumerate(ready_operations):
            if op["remaining_work"] == max_work:
                mo = op.get("machine_options", {})
                if mo:
                    best_machine = min(mo, key=lambda m: (_machine_loads.get(m, 0), mo[m]))
                    return (i, best_machine)
                return i
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # Soft-min of machine durations
    min_durations = np.zeros(n, dtype=float)
    # Also track the second-best duration to gauge flexibility
    second_durations = np.full(n, np.inf, dtype=float)
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            vals = sorted(mo.values())
            if len(vals) >= 2:
                min_durations[i] = vals[0]
                second_durations[i] = vals[1]
            else:
                min_durations[i] = vals[0]
        else:
            min_durations[i] = 0.0

    num_options = np.array([
        len(op["machine_options"]) if op["machine_options"] else 1
        for op in ready_operations
    ], dtype=float)

    num_options_safe = np.maximum(num_options, 1)
    flexibility = 1.0 / num_options_safe

    # Work density
    remaining_ops_safe = np.maximum(remaining_ops, 1)
    work_density = remaining_works / remaining_ops_safe

    # Machine load penalty: for each operation, compute the minimum load
    # among its available machines. High load = worse choice.
    machine_load_penalties = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            loads = np.array([_machine_loads.get(m, 0) for m in mo.keys()], dtype=float)
            machine_load_penalties[i] = np.min(loads)
        else:
            machine_load_penalties[i] = 0.0

    # Normalize features for stable weighting
    rw_max = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    md_max = np.max(min_durations) if np.max(min_durations) > 0 else 1.0
    ml_max = np.max(machine_load_penalties) if np.max(machine_load_penalties) > 0 else 1.0

    rw_norm = remaining_works / rw_max
    md_norm = min_durations / md_max
    ml_norm = machine_load_penalties / ml_max

    # Adaptive weights: as simulation progresses, emphasize load balancing more
    progress = min(_dispatch_count / 50.0, 1.0)  # ramps up over first 50 dispatches
    w_work = 1.0
    w_dur = 0.10 + 0.10 * progress
    w_density = 0.25
    w_flex = 0.05
    w_load = 0.15 + 0.25 * progress  # increasing importance of load balancing

    scores = (
        w_work * rw_norm
        - w_dur * md_norm
        + w_density * work_density
        + w_flex * flexibility
        - w_load * ml_norm
    )

    # Small look-ahead: penalize ops that have very few machine options
    # AND those machines are heavily loaded (trapped ops)
    trapped_penalty = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo and len(mo) == 1:
            m = list(mo.keys())[0]
            load = _machine_loads.get(m, 0)
            if load > 0:
                trapped_penalty[i] = load / (ml_max + 1e-9) * 0.3
    scores -= trapped_penalty

    best_idx = int(np.argmax(scores))

    # Select machine: least loaded, tie-break by shortest duration
    best_op = ready_operations[best_idx]
    machine_options = best_op.get("machine_options", {})
    if machine_options:
        best_machine = min(
            machine_options,
            key=lambda m: (_machine_loads.get(m, 0), machine_options[m])
        )
        return (best_idx, best_machine)

    return best_idx
# EVOLVE-BLOCK-END
