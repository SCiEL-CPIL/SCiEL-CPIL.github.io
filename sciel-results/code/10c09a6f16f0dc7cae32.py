"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 145.0  # safety margin before 150s kill
_START_TIME = _time.time()

# Track machine loads to identify bottlenecks
_machine_load = {}


def _ensure_deadline():
    """Return True if we are within the wall-clock budget."""
    return (_time.time() - _START_TIME) < _DEADLINE


def _update_machine_loads(ready_operations, time):
    """
    Update machine load estimates based on ready operations.
    Weight by duration to identify which machines are becoming bottlenecks.
    Also track how many operations can use each machine (flexibility signal).
    """
    # Reset loads for this dispatch call
    _machine_load.clear()
    
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine, duration in machine_options.items():
            if machine not in _machine_load:
                _machine_load[machine] = 0.0
            # Weight by duration: longer ops create more congestion
            _machine_load[machine] += duration


def _compute_op_score(op, time):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).
    
    Uses machine-load balancing strategy with soft-min for duration selection
    and stronger congestion-aware penalties to prevent bottlenecks.
    """
    work = op["remaining_work"]
    machine_options = op["machine_options"]
    remaining_ops = op.get("remaining_ops", 0)

    if not machine_options:
        return 0.0

    durations = np.array(list(machine_options.values()), dtype=float)
    machines = list(machine_options.keys())
    num_machines = len(machines)

    # Soft-min for duration (smooth surrogate for min): lower = more urgent
    # Use temperature to control sharpness
    temp = 1.0
    log_sum_exp = np.log(np.sum(np.exp(durations / temp)))
    soft_min_dur = temp * (np.log(num_machines) - log_sum_exp)  # approximates min
    max_dur = float(np.max(durations))
    min_dur = float(np.min(durations))

    # Machine congestion: how loaded are the candidate machines?
    machine_loads = np.array([_machine_load.get(m, 0.0) for m in machines], dtype=float)
    min_load = float(np.min(machine_loads))
    max_load = float(np.max(machine_loads))
    avg_load = float(np.mean(machine_loads))

    # Global context: what's the overall max load across all machines?
    all_loads = np.array(list(_machine_load.values()), dtype=float) if _machine_load else np.array([0.0])
    global_max_load = float(np.max(all_loads)) if len(all_loads) > 0 else 0.0

    # Score components:
    # 1. Remaining work (primary): more work = higher priority
    score = 1.0 * work

    # 2. Machine load balancing: strong bonus if this op can use a less-loaded machine
    #    Use soft-min of loads for smoothness
    if avg_load > 0:
        load_temp = 1.0
        log_sum_exp_load = np.log(np.sum(np.exp(machine_loads / load_temp)))
        soft_min_load = load_temp * (np.log(num_machines) - log_sum_exp_load)
        # Normalize: how much better is this op's best option vs global max?
        congestion_ratio = soft_min_load / (global_max_load + 1e-9)
        score += 1.5 * (1.0 - congestion_ratio)  # Stronger weight for congestion avoidance

    # 3. Penalty if ALL machines for this op are heavily congested
    #    (no escape from the bottleneck)
    if global_max_load > 0:
        all_congested = float(np.mean(machine_loads / (global_max_load + 1e-9)))
        if all_congested > 0.7:  # Most options are on busy machines
            score -= 0.5 * all_congested

    # 4. Flexibility: more machine options is better (log-scaled)
    score += 0.3 * np.log1p(num_machines)

    # 5. Duration urgency: shorter operations finish faster
    #    Use soft-min for smoothness
    score += 0.4 / (soft_min_dur + 1e-9)

    # 6. Urgency: fewer remaining ops = closer to completion
    score += 0.3 / (remaining_ops + 1e-9)

    # 7. Penalty for high duration variance (uncertainty in completion)
    dur_spread = (max_dur - min_dur) / (min_dur + 1e-9)
    score -= 0.15 * dur_spread

    return score


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Returns: index of the ready operation to start.
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a fast rule
    if not _ensure_deadline():
        # Fast fallback: pick the operation with the most remaining work
        best_idx = 0
        best_work = ready_operations[0]["remaining_work"]
        for i, op in enumerate(ready_operations[1:], 1):
            if op["remaining_work"] > best_work:
                best_work = op["remaining_work"]
                best_idx = i
        return best_idx

    # Update machine load estimates
    _update_machine_loads(ready_operations, time)

    # Compute scores for all ready operations
    scores = np.array([_compute_op_score(op, time) for op in ready_operations], dtype=float)

    # Add tiny random tiebreaker to avoid systematic tie-breaking
    # that can cause pathological scheduling patterns
    if len(ready_operations) > 1:
        scores += np.random.uniform(0, 1e-6, size=len(scores))

    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
