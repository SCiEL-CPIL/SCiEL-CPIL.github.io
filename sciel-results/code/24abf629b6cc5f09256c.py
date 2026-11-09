"""Job-Shop Scheduling dispatcher with dynamic bottleneck identification and multi-criteria scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill


def _deadline_exceeded():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a fast simple rule
    if _deadline_exceeded():
        # Fast fallback: pick operation with smallest min duration (SPT)
        min_durs = []
        for op in ready_operations:
            mo = op.get("machine_options", {})
            if mo:
                min_durs.append(min(mo.values()))
            else:
                min_durs.append(1.0)
        return int(np.argmin(min_durs))

    # Extract features
    remaining_work = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    max_durations = np.empty(n, dtype=np.float64)
    avg_durations = np.empty(n, dtype=np.float64)
    machine_options_list = []

    for i, op in enumerate(ready_operations):
        remaining_work[i] = op.get("remaining_work", 0.0)
        remaining_ops[i] = op.get("remaining_ops", 1.0)
        mo = op.get("machine_options", {})
        if mo:
            durs = list(mo.values())
            min_durations[i] = min(durs)
            max_durations[i] = max(durs)
            avg_durations[i] = sum(durs) / len(durs)
            machine_options_list.append(mo)
        else:
            min_durations[i] = 1.0
            max_durations[i] = 1.0
            avg_durations[i] = 1.0
            machine_options_list.append({0: 1.0})

    # --- Dynamic bottleneck identification + multi-criteria scoring ---
    
    # Build machine load estimates: for each machine, sum of durations of all ready ops that can use it
    all_machines = set()
    for mo in machine_options_list:
        all_machines.update(mo.keys())
    
    if not all_machines:
        all_machines = {0}
    
    machine_load = {m: 0.0 for m in all_machines}
    for i, mo in enumerate(machine_options_list):
        for m, dur in mo.items():
            machine_load[m] += dur

    # For each operation, find the best machine (least loaded + shortest duration)
    # and compute its estimated completion time
    op_completion_times = np.empty(n, dtype=np.float64)
    op_best_machines = [None] * n

    for i in range(n):
        mo = machine_options_list[i]
        best_m = None
        best_ct = float('inf')
        for m, dur in mo.items():
            ct = machine_load[m] + dur
            if ct < best_ct:
                best_ct = ct
                best_m = m
        op_completion_times[i] = best_ct
        op_best_machines[i] = best_m

    # Compute critical path reduction score: for each candidate i, estimate how much
    # scheduling it reduces the maximum completion time among remaining operations
    critical_scores = np.empty(n, dtype=np.float64)
    
    # Pre-compute current max completion time
    current_max_ct = float(np.max(op_completion_times))
    
    for i in range(n):
        m_i = op_best_machines[i]
        dur_i = machine_options_list[i].get(m_i, 1.0)
        
        # New machine load for m_i after scheduling i
        new_load_m_i = machine_load[m_i] - dur_i
        
        # For each other operation j, compute new completion time
        max_ct = 0.0
        for j in range(n):
            if j == i:
                continue
            m_j = op_best_machines[j]
            dur_j = machine_options_list[j].get(m_j, 1.0)
            if m_j == m_i:
                new_ct = new_load_m_i + dur_j
            else:
                new_ct = op_completion_times[j]
            if new_ct > max_ct:
                max_ct = new_ct
        
        # Lower max_ct is better; use negative for maximization
        critical_scores[i] = -max_ct

    # --- Multi-criteria scoring ---
    def _norm(arr):
        mn = arr.min()
        mx = arr.max()
        if mx <= mn + 1e-12:
            return np.full_like(arr, 0.5, dtype=np.float64)
        return (arr - mn) / (mx - mn)

    nw = _norm(remaining_work)          # higher is better (more urgent)
    no_inv = 1.0 - _norm(remaining_ops)  # lower remaining_ops is better
    nd = 1.0 - _norm(min_durations)     # lower duration is better (SPT)
    nd_avg = 1.0 - _norm(avg_durations)  # lower avg duration is better

    # Work per remaining op: higher means heavier ops remaining
    with np.errstate(divide='ignore', invalid='ignore'):
        work_per_op = remaining_work / np.maximum(remaining_ops, 1.0)
    npw = _norm(work_per_op)  # higher is better (heavier ops = more critical)

    # Normalize critical scores (higher is better)
    ncrit = _norm(critical_scores)

    # Weighted combination
    # Emphasize: critical path reduction, remaining work urgency, closeness to completion, SPT
    w_crit = 0.35
    w_work = 0.25
    w_close = 0.20
    w_spt = 0.20

    score = (w_crit * ncrit +
             w_work * nw +
             w_close * no_inv +
             w_spt * nd)

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
