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

    # --- Greedy Lookahead Simulation: minimize projected makespan ---
    # For each candidate operation i, simulate scheduling it (on its best machine),
    # then estimate the makespan of the remaining ready set.
    # Select the operation that minimizes the projected future horizon.
    
    all_machines = set()
    for mo in machine_options_list:
        all_machines.update(mo.keys())
    
    if not all_machines:
        all_machines = {0}
    
    # machine_load[m] = total work queued on machine m
    machine_load = {m: 0.0 for m in all_machines}
    for mo in machine_options_list:
        for m, dur in mo.items():
            machine_load[m] += dur

    # For each operation, pre-compute the best machine (least loaded)
    op_best_machine = [None] * n
    op_min_load = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        mo = machine_options_list[i]
        best_m = None
        best_load = float('inf')
        for m, dur in mo.items():
            if machine_load[m] < best_load:
                best_load = machine_load[m]
                best_m = m
        op_best_machine[i] = best_m
        op_min_load[i] = best_load

    # Greedy lookahead: for each candidate i, simulate scheduling it,
    # then estimate the makespan of the remaining operations.
    projected_makespan = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        m_i = op_best_machine[i]
        dur_i = machine_options_list[i].get(m_i, 1.0)
        
        # After scheduling i, update machine loads
        # i is removed from the queue, so its duration is subtracted from m_i's load
        new_load_m_i = machine_load[m_i] - dur_i
        
        # Estimate makespan of remaining operations
        # For each remaining operation j, estimate its completion time
        # assuming it runs on its best machine after the current work is done
        
        max_completion = 0.0
        
        for j in range(n):
            if j == i:
                continue
            m_j = op_best_machine[j]
            dur_j = machine_options_list[j].get(m_j, 1.0)
            
            # Estimate start time of j on machine m_j
            # If m_j == m_i, the work ahead is new_load_m_i - dur_j (excluding j itself)
            # If m_j != m_i, the work ahead is machine_load[m_j] - dur_j
            if m_j == m_i:
                work_ahead = max(0.0, new_load_m_i - dur_j)
            else:
                work_ahead = max(0.0, machine_load[m_j] - dur_j)
            
            # Estimated completion time = work_ahead + dur_j
            completion_j = work_ahead + dur_j
            
            if completion_j > max_completion:
                max_completion = completion_j
        
        # Add the completion time of operation i itself
        # i completes at dur_i (it starts now on m_i)
        if dur_i > max_completion:
            max_completion = dur_i
        
        projected_makespan[i] = max_completion

    # Select the operation that minimizes projected makespan
    best_idx = int(np.argmin(projected_makespan))
    
    # Tie-breaking: if multiple operations have similar projected makespan,
    # use secondary criteria to break ties
    min_makespan = projected_makespan.min()
    close_to_min = np.where(projected_makespan <= min_makespan + 1e-9)[0]
    
    if len(close_to_min) > 1:
        # Use WSPT + closeness to completion as tie-breakers
        def _norm(arr):
            mn = arr.min()
            mx = arr.max()
            if mx <= mn + 1e-12:
                return np.full_like(arr, 0.5, dtype=np.float64)
            return (arr - mn) / (mx - mn)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            wspt = remaining_work / np.maximum(min_durations, 1e-6)
        n_wspt = _norm(wspt)
        n_close = 1.0 - _norm(remaining_ops)
        n_spt = 1.0 - _norm(min_durations)
        
        # Score only among tied candidates
        tie_score = (0.5 * n_wspt[close_to_min] + 
                     0.3 * n_close[close_to_min] + 
                     0.2 * n_spt[close_to_min])
        
        best_tie_idx = int(np.argmax(tie_score))
        best_idx = int(close_to_min[best_tie_idx])
    
    return best_idx
# EVOLVE-BLOCK-END
