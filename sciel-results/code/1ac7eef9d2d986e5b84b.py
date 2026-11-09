"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin before 150s kill

def _deadline_exceeded():
    return _time.monotonic() >= _DEADLINE

def _soft_min(values, alpha=2.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation of min."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    if len(values) == 1:
        return values[0]
    neg_alpha_vals = -alpha * values
    max_neg = np.max(neg_alpha_vals)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_alpha_vals - max_neg)))
    return -log_sum / alpha

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling using dynamic bottleneck identification.
    Selects the operation whose assignment to its least-loaded available machine
    maximizes the minimum slack time across all currently ready operations,
    thereby actively preventing machine starvation and reducing idle gaps.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, collect machine options as list of (machine, duration)
    machine_opts_list = []
    for op in ready_operations:
        if op["machine_options"]:
            machine_opts_list.append(list(op["machine_options"].items()))
        else:
            machine_opts_list.append([("default", 1.0)])
    
    # Compute min and soft-min durations for each op
    min_durations = np.array([min(d for _, d in opts) if opts else 1.0 for opts in machine_opts_list], dtype=np.float64)
    soft_min_durations = np.array([
        _soft_min(np.array([d for _, d in opts], dtype=np.float64), alpha=3.0) if opts else 1.0
        for opts in machine_opts_list
    ], dtype=np.float64)

    # WSPT ratio: remaining_work / soft_min_duration
    wspt = remaining_work / np.maximum(soft_min_durations, 1e-9)
    
    # Build machine -> list of (op_index, duration) mapping
    machine_to_ops = {}
    for i, opts in enumerate(machine_opts_list):
        for machine, dur in opts:
            if machine not in machine_to_ops:
                machine_to_ops[machine] = []
            machine_to_ops[machine].append((i, dur))
    
    # For each machine, compute total pending load (sum of durations of all ready ops that can use it)
    machine_total_load = {}
    for machine, ops_list in machine_to_ops.items():
        total = 0.0
        for j, dur in ops_list:
            total += dur
        machine_total_load[machine] = total
    
    # For each candidate operation i, evaluate each possible machine assignment:
    # Compute the "minimum slack" across all other ready operations if i is assigned to machine m.
    # Slack for op j on machine m' (where m' != m or j != i): 
    #   If j can use machine m, its earliest start is delayed by dur_i if i uses m.
    #   We estimate the impact as: for each other op j, the minimum over its machines of
    #   (duration_j + delay_if_conflict). The "slack" is the minimum across all j of this.
    #
    # Simplified approach: For candidate (i, m), compute for each other op j:
    #   - If j can use m: estimated completion = dur_j + dur_i (conflict)
    #   - Else: estimated completion = min duration of j's other machines
    #   - Slack_j = min over j's machines of (dur_jm + dur_i if m == m_conflict else dur_jm)
    #   - min_slack = min over all j of Slack_j
    # We want to maximize min_slack (more slack = better).
    #
    # But this is O(n^2 * machines). For n=10, this is fine.
    
    # For each candidate i, find the best machine assignment that maximizes min slack
    # among all other ready ops, weighted by urgency.
    
    # Precompute for each op j, the min duration over machines excluding a given machine
    # Actually, let's just compute directly for each (i, m) pair.
    
    best_score = -np.inf
    best_idx = 0
    best_machine = None
    
    for i in range(n):
        opts_i = machine_opts_list[i]
        if not opts_i:
            continue
        
        for m_i, dur_i in opts_i:
            # Estimate min slack across all other ops if i uses machine m_i
            min_slack = np.inf
            
            for j in range(n):
                if j == i:
                    continue
                opts_j = machine_opts_list[j]
                if not opts_j:
                    continue
                
                # For op j, compute the minimum estimated completion time
                # considering that machine m_i is occupied for dur_i
                min_est_completion = np.inf
                for m_j, dur_j in opts_j:
                    if m_j == m_i:
                        # Conflict: j must wait for i on machine m_i
                        est_completion = dur_j + dur_i
                    else:
                        # No conflict: j can use m_j immediately
                        est_completion = dur_j
                    if est_completion < min_est_completion:
                        min_est_completion = est_completion
                
                if min_est_completion < min_slack:
                    min_slack = min_est_completion
            
            if min_slack == np.inf:
                min_slack = 0.0
            
            # Score: combine min_slack (higher is better) with urgency (WSPT)
            # We want to pick the op that creates the most slack for others,
            # but also prioritize urgent ops.
            # Normalize min_slack across all (i, m) candidates later.
            # For now, use a combined score: urgency * (1 + normalized_slack)
            
            # Actually, let's collect all (i, m, min_slack, wspt) tuples and score them together
            # For now, store the min_slack for this (i, m)
            # We'll do a two-pass: first collect all, then score
    
    # Two-pass approach: collect all (i, m, min_slack) and then score
    candidates = []
    for i in range(n):
        opts_i = machine_opts_list[i]
        if not opts_i:
            continue
        
        for m_i, dur_i in opts_i:
            min_slack = np.inf
            
            for j in range(n):
                if j == i:
                    continue
                opts_j = machine_opts_list[j]
                if not opts_j:
                    continue
                
                min_est_completion = np.inf
                for m_j, dur_j in opts_j:
                    if m_j == m_i:
                        est_completion = dur_j + dur_i
                    else:
                        est_completion = dur_j
                    if est_completion < min_est_completion:
                        min_est_completion = est_completion
                
                if min_est_completion < min_slack:
                    min_slack = min_est_completion
            
            if min_slack == np.inf:
                min_slack = 0.0
            
            candidates.append((i, m_i, dur_i, min_slack, wspt[i]))
    
    if not candidates:
        return 0
    
    # Normalize min_slack across all candidates
    all_slacks = np.array([c[3] for c in candidates], dtype=np.float64)
    all_wspts = np.array([c[4] for c in candidates], dtype=np.float64)
    all_durs = np.array([c[2] for c in candidates], dtype=np.float64)
    
    mn_s = all_slacks.min()
    mx_s = all_slacks.max()
    if mx_s > mn_s:
        norm_slacks = (all_slacks - mn_s) / (mx_s - mn_s)
    else:
        norm_slacks = np.ones_like(all_slacks) * 0.5
    
    mn_w = all_wspts.min()
    mx_w = all_wspts.max()
    if mx_w > mn_w:
        norm_wspts = (all_wspts - mn_w) / (mx_w - mn_w)
    else:
        norm_wspts = np.ones_like(all_wspts) * 0.5
    
    mn_d = all_durs.min()
    mx_d = all_durs.max()
    if mx_d > mn_d:
        norm_durs = (all_durs - mn_d) / (mx_d - mn_d)
    else:
        norm_durs = np.ones_like(all_durs) * 0.5
    
    # Combined score: high slack is good, high urgency is good, short duration is good
    w_slack = 0.40
    w_wspt = 0.35
    w_spt = 0.25
    
    scores = w_slack * norm_slacks + w_wspt * norm_wspts + w_spt * (1.0 - norm_durs)
    
    best_candidate_idx = int(np.argmax(scores))
    best_i = candidates[best_candidate_idx][0]
    return best_i
# EVOLVE-BLOCK-END
