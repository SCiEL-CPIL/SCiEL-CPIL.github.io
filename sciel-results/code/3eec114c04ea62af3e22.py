"""Open-shop dispatch: Load-Balancing Water-Filling heuristic (refined)."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Track elapsed time to guard against the 150s wall-clock limit.
_start_time = _time.time()
_DEADLINE_FRACTION = 0.90  # safety margin: stop searching at 90% of budget
_DEADLINE_SECONDS = 150.0 * _DEADLINE_FRACTION


def _time_left():
    """Return seconds remaining before the safety deadline."""
    return _DEADLINE_SECONDS - (_time.time() - _start_time)


def _log_sum_exp(values):
    """Numerically stable log-sum-exp."""
    if not values:
        return 0.0
    m = max(values)
    if m == 0.0:
        return 0.0
    return m + math.log(sum(math.exp(x - m) for x in values))


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a Load-Balancing Water-Filling
    heuristic. For each candidate, choose the machine that minimizes the resulting
    smooth max load (log-sum-exp surrogate), then score by the resulting load
    profile plus remaining-work and duration terms.

    Returns:
        int index of the chosen ready operation.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Build a proxy machine load vector from all ready operations' durations.
    machine_load = {}
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            machine_load[m] = machine_load.get(m, 0.0) + dur

    if not machine_load:
        return 0

    max_load = max(machine_load.values())
    if max_load == 0:
        max_load = 1.0

    # Precompute normalized loads for efficiency.
    norm_loads = {m: load / max_load for m, load in machine_load.items()}
    machine_list = list(machine_load.keys())

    best_idx = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op["machine_options"]

        if not machine_options:
            continue

        # For each compatible machine, compute the smooth max load after assignment.
        # Choose the machine that minimizes this value (water-filling machine selection).
        best_machine_score = float('inf')
        best_machine = None
        best_dur = 0.0

        for m, dur in machine_options.items():
            # Simulate assigning this op to machine m.
            # New load for m: norm_loads[m] + dur/max_load
            # Other machines unchanged.
            # Compute log-sum-exp of all machine loads after assignment.
            values = []
            for mm in machine_list:
                if mm == m:
                    values.append(norm_loads[mm] + dur / max_load)
                else:
                    values.append(norm_loads[mm])
            lse = _log_sum_exp(values)
            if lse < best_machine_score:
                best_machine_score = lse
                best_machine = m
                best_dur = dur

        if best_machine is None:
            continue

        # Score: primary is the resulting smooth max load (lower is better).
        # Secondary: prefer operations with more remaining work (MWR) to avoid
        # starving long jobs.
        # Tertiary: prefer shorter processing time to free machines quickly.
        # Also consider remaining_ops: fewer remaining ops means the job is
        # nearly done, so completing it frees a job slot.
        remaining_work_norm = remaining_work / (max_load + 1e-9)
        dur_norm = best_dur / max_load
        ops_norm = float(remaining_ops) / (n + 1e-9)

        # Weights: lse dominates; remaining_work provides a mild pull toward
        # jobs with more total work; duration and remaining_ops provide tie-breaking.
        score = best_machine_score - 0.08 * remaining_work_norm + 0.04 * dur_norm + 0.02 * ops_norm

        if score < best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
