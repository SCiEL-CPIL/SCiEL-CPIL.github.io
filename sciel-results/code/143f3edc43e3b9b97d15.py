"""Open-Shop scheduling dispatch rule with tuned heuristic."""
# EVOLVE-BLOCK-START
import time as _time
import random

# ---- deadline guard -------------------------------------------------------
_T0 = _time.time()
_BUDGET = 130.0          # evaluator kills at 150s; keep 20s safety margin
_DEADLINE = _T0 + _BUDGET

# ---- tunable weights ------------------------------------------------------
# score = w_dur * duration + w_job * remaining_work + w_load * machine_free_time
# We'll tune these offline
_WEIGHTS = (1.0, 0.5, 0.3)  # initial guess


def _make_instance(n_jobs=10, n_machines=5, seed=0):
    """Generate a random open-shop instance."""
    rng = random.Random(seed)
    durations = [[rng.randint(1, 20) for _ in range(n_machines)] for _ in range(n_jobs)]
    return durations


def _simulate(durations, weights):
    """Event-driven open-shop simulation. Returns makespan."""
    n_jobs, n_machines = len(durations), len(durations[0])
    w_dur, w_job, w_load = weights

    # State
    remaining_ops = [set(range(n_machines)) for _ in range(n_jobs)]
    remaining_work = [sum(durations[j]) for j in range(n_jobs)]
    machine_free = [0.0] * n_machines
    job_free = [0.0] * n_jobs

    # All ops are ready initially in open-shop
    ready_ops = [(j, m) for j in range(n_jobs) for m in range(n_machines)]
    total_ops = n_jobs * n_machines
    done = 0
    makespan = 0.0

    while done < total_ops:
        # Find best operation to start
        best_key = None
        best_op = None
        best_start_t = 0.0

        for (j, m) in ready_ops:
            start_t = max(machine_free[m], job_free[j])
            dur = durations[j][m]
            machine_load = machine_free[m]
            # Higher score = better
            score = w_dur * dur + w_job * remaining_work[j] + w_load * machine_load
            # Primary: earliest start; Secondary: highest score
            key = (start_t, -score)
            if best_key is None or key < best_key:
                best_key = key
                best_op = (j, m)
                best_start_t = start_t

        if best_op is None:
            break

        j, m = best_op
        ready_ops.remove((j, m))
        end_t = best_start_t + durations[j][m]
        machine_free[m] = end_t
        job_free[j] = end_t
        remaining_ops[j].discard(m)
        remaining_work[j] -= durations[j][m]
        makespan = max(makespan, end_t)
        done += 1

    return makespan


def _tune_weights():
    """Tune weights on representative instances."""
    global _WEIGHTS
    grid = [
        (1.0, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (1.0, 0.0, 0.5),
        (1.0, 0.5, 0.5),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.8, 0.5, 0.5),
        (1.2, 0.3, 0.3),
        (1.0, 0.8, 0.2),
        (1.5, 0.0, 0.5),
        (0.5, 1.0, 0.5),
        (1.0, 0.2, 0.8),
    ]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    instances = [_make_instance(seed=s) for s in seeds]

    best_tuple = (1.0, 0.5, 0.3)
    best_obj = float('inf')

    for idx, pair in enumerate(grid):
        if _time.time() > _DEADLINE - 30.0:
            break
        total = 0.0
        for inst in instances:
            total += _simulate(inst, pair)
        obj = total / len(instances)
        if obj < best_obj:
            best_obj = obj
            best_tuple = pair

    _WEIGHTS = best_tuple
    return best_tuple


# Run tuning once at import time
_tune_weights()


def dispatch(ready_operations, time):
    """Pick which ready operation to start next."""
    w_dur, w_job, w_load = _WEIGHTS
    best_idx = 0
    best_score = -1.0
    n = len(ready_operations)

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        # Choose the machine that gives best score
        best_machine_score = -1.0
        best_duration = 0
        for machine, duration in opts.items():
            # Estimate machine load: we don't have direct access, use duration as proxy
            # In practice, the simulator will handle actual machine free times
            score = w_dur * duration + w_job * op["remaining_work"]
            if score > best_machine_score:
                best_machine_score = score
                best_duration = duration

        # Use the best machine's score for this operation
        if best_machine_score > best_score:
            best_score = best_machine_score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
