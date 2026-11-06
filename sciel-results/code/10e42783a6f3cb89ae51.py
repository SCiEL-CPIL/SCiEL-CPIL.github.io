"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = _time.time() + 145.0  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - _time.time()


def _normalize(arr):
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Deadline guard: fall back to a fast, cheap rule near the limit.
    if _time_left() < 0.5:
        durations = []
        for op in ready_operations:
            mo = op["machine_options"]
            if mo:
                durations.append(min(mo.values()))
            else:
                durations.append(float('inf'))
        return int(np.argmin(durations))

    # ---- Gather per-operation metrics ----
    remaining_works = []
    min_durations = []
    remaining_ops_counts = []
    all_machines = set()
    op_opts = []

    for op in ready_operations:
        mo = op["machine_options"]
        op_opts.append(mo)
        if mo:
            min_durations.append(min(mo.values()))
        else:
            min_durations.append(float('inf'))
        all_machines.update(mo.keys())
        remaining_works.append(op["remaining_work"])
        remaining_ops_counts.append(op["remaining_ops"])

    remaining_works = np.array(remaining_works, dtype=np.float64)
    min_durations = np.array(min_durations, dtype=np.float64)
    remaining_ops_counts = np.array(remaining_ops_counts, dtype=np.float64)

    norm_rw = _normalize(remaining_works)
    norm_md = _normalize(min_durations)
    norm_ro = _normalize(remaining_ops_counts)

    # ---- Machine load model (for the smooth load-balance surrogate) ----
    # For each machine, estimate the total work that would land on it if every
    # ready op were assigned to its best (shortest) machine, weighted by how
    # "attractive" that machine is for the op.
    machine_loads = {m: 0.0 for m in all_machines}
    for i in range(n):
        mo = op_opts[i]
        if not mo:
            continue
        best_dur = min_durations[i]
        for m, dur in mo.items():
            if dur > 0:
                weight = best_dur / dur
            else:
                weight = 1.0
            machine_loads[m] += dur * weight

    if all_machines:
        load_values = np.array(list(machine_loads.values()), dtype=np.float64)
        current_mean_load = float(np.mean(load_values))
    else:
        current_mean_load = 0.0

    # ---- Smooth load-balance surrogate via log-sum-exp ----
    # For each candidate op, we want a smooth measure of how much "extra" load
    # it would push onto the machines it can use (above the mean). Instead of
    # the nonsmooth max/argmin, use a temperature-scaled log-sum-exp (soft-max)
    # over the candidate's compatible machines. Lower = better balance.
    LSE_T = 1.0  # temperature: lower -> sharper (closer to hard max)
    load_balance_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mo = op_opts[i]
        if not mo:
            load_balance_scores[i] = 0.0
            continue
        # soft-max of (current_load + dur - mean) over compatible machines
        vals = np.array(
            [machine_loads.get(m, 0.0) + mo[m] - current_mean_load for m in mo.keys()],
            dtype=np.float64,
        )
        scaled = vals / LSE_T
        # numerically stable log-sum-exp
        mmax = np.max(scaled)
        lse = mmax + np.log(np.sum(np.exp(scaled - mmax)))
        # convert back to original scale (this is a smooth max, not log)
        load_balance_scores[i] = LSE_T * lse

    # Normalize the load-balance term so it's comparable to the others.
    norm_lb = _normalize(load_balance_scores)

    # ---- Combined weighted score ----
    # Prioritize:
    #   1. Most remaining job work (critical path)      -> high
    #   2. Shorter processing time (SPT)                -> low min_dur
    #   3. More remaining ops (urgency)                 -> high
    #   4. Better load balance (smooth surrogate)       -> low
    score = (
        0.45 * norm_rw           # most remaining job work first
        + 0.20 * (1.0 - norm_md)  # shorter processing time first
        + 0.15 * norm_ro         # more remaining ops = more urgent
        + 0.20 * (1.0 - norm_lb)  # smoother load balance first
    )

    best_idx = int(np.argmax(score))

    # For the chosen operation, pick the machine that minimizes the smooth
    # load-increase (consistent with the surrogate objective used in scoring).
    mo = op_opts[best_idx]
    if mo:
        best_m = min(mo.keys(), key=lambda m: (machine_loads.get(m, 0.0) + mo[m], m))
        return (best_idx, best_m)
    return best_idx
# EVOLVE-BLOCK-END
