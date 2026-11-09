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

    # --- Dynamic bottleneck identification via minimum-slack maximization ---
    # Strategy: For each candidate operation i, simulate assigning it to its least-loaded
    # available machine, then compute the minimum slack across all remaining ready operations.
    # Select the operation that maximizes this minimum slack, preventing machine starvation.
    
    # Build machine load estimates
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

    # For each operation, pre-compute the best machine (least loaded) and its position in queue
    # op_best_machine[i] = machine with least load that op i can use
    # op_queue_pos[i, m] = estimated position (load before this op) on machine m
    op_best_machine = [None] * n
    op_min_load = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        mo = machine_options_list[i]
        best_m = None
        best_load = float('inf')
        for m, dur in mo.items():
            # The load on machine m excluding this op's contribution
            # Actually, machine_load includes this op. The "queue ahead" is machine_load[m] - dur
            # But we want to pick the machine where the op would be least constrained
            # Use the current total load as proxy for congestion
            if machine_load[m] < best_load:
                best_load = machine_load[m]
                best_m = m
        op_best_machine[i] = best_m
        op_min_load[i] = best_load

    # For each candidate i, compute the soft-min slack after scheduling i
    # Using log-sum-exp for a smooth surrogate of the min operation
    # This provides a smoother scoring signal that better distinguishes candidates
    
    # Temperature parameter for soft-min: lower = closer to hard min, higher = softer
    softmin_temp = 5.0
    
    min_slack_scores = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        m_i = op_best_machine[i]
        dur_i = machine_options_list[i].get(m_i, 1.0)
        
        # After scheduling i, the load on m_i that remains for others is:
        new_load_m_i = machine_load[m_i] - dur_i
        
        # Collect slack values for all other operations
        slacks = []
        for j in range(n):
            if j == i:
                continue
            m_j = op_best_machine[j]
            dur_j = machine_options_list[j].get(m_j, 1.0)
            
            if m_j == m_i:
                slack_j = new_load_m_i - dur_j
            else:
                slack_j = machine_load[m_j] - dur_j
            
            # Clamp to non-negative
            if slack_j < 0:
                slack_j = 0.0
            
            slacks.append(slack_j)
        
        if not slacks:
            min_slack_scores[i] = 0.0
            continue
        
        slacks_arr = np.array(slacks, dtype=np.float64)
        
        # Soft-min via log-sum-exp: softmin(x) = -logsumexp(-x/temp) * temp
        # This is a smooth approximation of min(x)
        neg_slacks = -slacks_arr / softmin_temp
        max_neg = np.max(neg_slacks)
        if max_neg == float('-inf'):
            min_slack_scores[i] = 0.0
        else:
            log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_slacks - max_neg)))
            soft_min_val = -log_sum_exp * softmin_temp
            # Clamp to non-negative
            if soft_min_val < 0:
                soft_min_val = 0.0
            min_slack_scores[i] = soft_min_val

    # --- Secondary criteria to break ties and add robustness ---
    def _norm(arr):
        mn = arr.min()
        mx = arr.max()
        if mx <= mn + 1e-12:
            return np.full_like(arr, 0.5, dtype=np.float64)
        return (arr - mn) / (mx - mn)

    # WSPT: work per unit time on the operation (urgency signal)
    with np.errstate(divide='ignore', invalid='ignore'):
        wspt = remaining_work / np.maximum(min_durations, 1e-6)
    n_wspt = _norm(wspt)  # higher = more urgent

    # Closeness to completion: fewer remaining ops = closer to done
    n_close = 1.0 - _norm(remaining_ops)

    # SPT: shorter min duration preferred
    n_spt = 1.0 - _norm(min_durations)

    # Normalize slack scores (higher is better)
    n_slack = _norm(min_slack_scores)

    # Weighted combination: prioritize slack maximization, then urgency
    # Adjusted weights: slightly reduce slack weight to let other criteria have more influence
    # since soft-min is already a smoothed version
    w_slack = 0.45
    w_wspt = 0.28
    w_close = 0.17
    w_spt = 0.10

    score = (w_slack * n_slack +
             w_wspt * n_wspt +
             w_close * n_close +
             w_spt * n_spt)

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
