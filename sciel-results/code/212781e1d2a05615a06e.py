"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

# Track machine load to prefer less-busy machines
_machine_load = {}
_job_remaining_work = {}
_job_remaining_ops = {}


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: simple urgency-based dispatch
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            # Pick the fastest machine; tie-break by least loaded
            best_machine = min(
                machine_options,
                key=lambda m: (machine_options[m], _machine_load.get(m, 0))
            )
            # Update load tracking
            _machine_load[best_machine] = _machine_load.get(best_machine, 0) + machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # For each operation, compute the best (min) duration
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)

    # Soft-min via log-sum-exp for a smoother "effective speed" signal
    # Lower soft-min duration => faster operation
    log_durations = np.log(min_durations + 1e-9)
    # Use a temperature-scaled soft-min: as T->0, approaches true min
    # But since we already have min_durations, let's use a different smooth approach:
    # effective_speed = 1 / (1 + 0.5 * (avg - min)) to blend speed and flexibility
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)

    # Normalize
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_ops = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0
    max_inv_min_dur = np.max(1.0 / min_durations) if np.any(min_durations > 0) else 1.0

    # Normalize each feature to [0, 1]
    norm_work = remaining_works / (max_work + 1e-9)
    norm_ops = remaining_ops / (max_ops + 1e-9)
    norm_speed = (1.0 / min_durations) / (max_inv_min_dur + 1e-9)

    # Weighted scoring:
    # - Urgency (remaining work): critical path length
    # - Depth (remaining ops): number of future ops
    # - Speed (fastest machine): prefer short processing times
    w_work = 1.0
    w_ops = 0.35
    w_speed = 0.55

    scores = (w_work * norm_work +
              w_ops * norm_ops +
              w_speed * norm_speed)

    best_idx = int(np.argmax(scores))

    # Choose the fastest eligible machine for the selected operation;
    # tie-break by least-loaded to balance
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        best_machine = min(
            machine_options,
            key=lambda m: (machine_options[m], _machine_load.get(m, 0))
        )
        # Update load tracking
        _machine_load[best_machine] = _machine_load.get(best_machine, 0) + machine_options[best_machine]
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
