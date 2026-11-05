"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

# Global state for load balancing: track cumulative processing time assigned to each machine
_machine_load = {}


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
            # Choose machine with lowest projected completion time
            best_machine = min(machine_options, key=lambda m: machine_options[m] + _machine_load.get(m, 0.0))
            _machine_load[best_machine] = _machine_load.get(best_machine, 0.0) + machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Extract key features for all ready operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ])

    # Compute critical path-like estimate: remaining work divided by number of remaining ops
    avg_work_per_op = np.where(remaining_ops > 0, remaining_works / np.maximum(remaining_ops, 1), remaining_works)

    # Compute effective duration using soft-min (log-sum-exp) for smooth machine selection
    beta = 5.0
    soft_min_durations = np.zeros(n)
    for i in range(n):
        durations = np.array(list(ready_operations[i]["machine_options"].values()), dtype=float)
        if len(durations) > 0:
            log_sum_exp = np.log(np.sum(np.exp(-beta * durations)))
            soft_min_durations[i] = -log_sum_exp / beta
        else:
            soft_min_durations[i] = np.inf

    # Use soft-min duration for efficiency scoring
    inv_soft_durations = 1.0 / np.maximum(soft_min_durations, 1e-9)

    # Compute machine contention: how many ready operations want each machine
    machine_counts = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    # For each operation, compute the minimum contention among its machine options
    min_contention = np.zeros(n)
    for i in range(n):
        if ready_operations[i]["machine_options"]:
            min_contention[i] = min(machine_counts.get(m, 0) for m in ready_operations[i]["machine_options"])
        else:
            min_contention[i] = 0

    # NEW: Compute projected completion time on least-loaded eligible machine
    # This is the key load-balancing signal: prefer ops that can finish sooner
    # given current machine loads
    projected_completion = np.zeros(n)
    for i in range(n):
        mo = ready_operations[i]["machine_options"]
        if mo:
            # Find the machine with minimum (current_load + duration)
            min_proj = min(_machine_load.get(m, 0.0) + mo[m] for m in mo)
            projected_completion[i] = min_proj
        else:
            projected_completion[i] = np.inf

    # Inverse projected completion: lower projected time = higher priority
    # Use soft inverse to avoid division by zero
    inv_projected = 1.0 / np.maximum(projected_completion, 1e-9)

    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_avg = np.max(avg_work_per_op) if np.max(avg_work_per_op) > 0 else 1.0
    max_inv_dur = np.max(inv_soft_durations) if np.any(inv_soft_durations > 0) else 1.0
    max_contention = np.max(min_contention) if np.max(min_contention) > 0 else 1.0
    max_inv_proj = np.max(inv_projected) if np.any(inv_projected > 0) else 1.0

    # Normalize components to [0, 1]
    normalized_urgency = remaining_works / (max_work + 1e-9)
    normalized_avg = avg_work_per_op / (max_avg + 1e-9)
    normalized_efficiency = inv_soft_durations / (max_inv_dur + 1e-9)
    normalized_low_contention = 1.0 - (min_contention / (max_contention + 1e-9))
    normalized_load_balance = inv_projected / (max_inv_proj + 1e-9)

    # Normalize remaining_ops: fewer remaining ops = closer to completion = higher priority
    max_remaining_ops = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0
    normalized_complete = 1.0 - (remaining_ops / (max_remaining_ops + 1e-9))

    # Weighted score combining:
    # - Urgency: total remaining work (critical path length proxy)
    # - Depth-adjusted urgency: average work per remaining op
    # - Efficiency: how fast the operation can be completed (soft-min over machine options)
    # - Completeness: jobs closer to completion get a small boost
    # - Low contention: prefer ops with less machine congestion
    # - Load balance: prefer ops that can finish sooner on least-loaded machine
    w_urgency = 1.2
    w_avg = 0.4
    w_efficiency = 0.4
    w_complete = 0.2
    w_contention = 0.3
    w_load_balance = 1.0

    scores = (w_urgency * normalized_urgency + w_avg * normalized_avg +
              w_efficiency * normalized_efficiency + w_complete * normalized_complete +
              w_contention * normalized_low_contention + w_load_balance * normalized_load_balance)

    best_idx = int(np.argmax(scores))

    # Choose the best machine for the selected operation.
    # Use a composite machine score: prefer lower projected completion time (load + duration)
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        best_machine = min(machine_options, key=lambda m: (
            _machine_load.get(m, 0.0) + machine_options[m] + 0.1 * machine_counts.get(m, 0),
            m
        ))
        # Update global machine load
        _machine_load[best_machine] = _machine_load.get(best_machine, 0.0) + machine_options[best_machine]
        return (best_idx, best_machine)
    return best_idx


def _soft_min_value(values, beta=5.0):
    """Compute soft-min using log-sum-exp for numerical stability."""
    if len(values) == 0:
        return np.inf
    arr = np.array(values, dtype=float)
    min_val = np.min(arr)
    shifted = arr - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - log_sum / beta
# EVOLVE-BLOCK-END
