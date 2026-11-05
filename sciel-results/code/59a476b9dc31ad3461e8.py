"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 140.0
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: Combines Bottom Level criticality with soft-max resource contention
    penalty and dynamic simulation-based bottleneck detection.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency fallback
        durations = np.asarray(instance["durations"], dtype=float)
        if len(durations) == 0: return np.array([])
        return durations / (np.max(durations) + 1e-9)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [None] * n)

    # 1. Build graph
    pred_lists = []
    for p in preds:
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                successors[p].append(i)
                in_degree[i] += 1
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) < n:
        top_order = list(range(n))

    # 2. Compute Bottom Level (Longest path from node to end)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            max_succ_bl = np.max([bottom_level[v] for v in successors[u]])
        else:
            max_succ_bl = 0.0
        bottom_level[u] = durations[u] + max_succ_bl

    # 3. Compute ES/EF and TF for reference
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        if pred_lists[u]:
            es[u] = np.max([ef[p] for p in pred_lists[u]])
        else:
            es[u] = 0.0
        ef[u] = es[u] + durations[u]
        
    cp_length = np.max(ef) if n > 0 else 0.0
    
    ls = np.zeros(n)
    lf = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            lf[u] = np.min([ls[v] for v in successors[u]])
        else:
            lf[u] = cp_length
        ls[u] = lf[u] - durations[u]
        
    tf = ls - es

    # 4. Parse Demands and Capacities
    n_res = 0
    d_matrix = None
    capacities_arr = None
    
    if demands is not None and capacities is not None:
        capacities_arr = np.asarray(capacities, dtype=float)
        if len(capacities_arr) == 0:
            capacities_arr = np.array([1.0])
        else:
            capacities_arr = np.where(capacities_arr > 0, capacities_arr, 1.0)
        n_res = len(capacities_arr)
        
        d_matrix = np.zeros((n, n_res), dtype=float)
        
        if isinstance(demands, list):
            for i in range(n):
                if i < len(demands):
                    d_i = demands[i]
                    if isinstance(d_i, dict):
                        for r, val in d_i.items():
                            if r < n_res:
                                d_matrix[i, r] = val
                    elif isinstance(d_i, (list, tuple, np.ndarray)):
                        d_arr = np.asarray(d_i, dtype=float)
                        if d_arr.ndim == 1:
                            min_len = min(len(d_arr), n_res)
                            d_matrix[i, :min_len] = d_arr[:min_len]
                    else:
                        if n_res > 0:
                            d_matrix[i, 0] = float(d_i)
        else:
            try:
                d_mat_full = np.asarray(demands, dtype=float)
                if d_mat_full.shape[0] == n:
                    if d_mat_full.ndim == 1:
                         d_matrix[:, 0] = d_mat_full
                    else:
                        min_res = min(d_mat_full.shape[1], n_res)
                        d_matrix[:, :min_res] = d_mat_full[:, :min_res]
            except:
                pass

    # 5. Simulate Resource-Feasible Greedy Schedule to identify bottlenecks
    sim_start = np.zeros(n)
    sim_finish = np.zeros(n)
    
    if d_matrix is not None and capacities_arr is not None and n_res > 0:
        remaining = set(range(n))
        scheduled = set()
        active = []  # list of (finish_time, activity_id)
        
        sim_priority = bottom_level.copy()
        
        while len(scheduled) < n:
            eligible = [i for i in remaining if all(p in scheduled for p in pred_lists[i])]
            if not eligible:
                break
            
            best_i = max(eligible, key=lambda x: sim_priority[x])
            
            earliest_start = 0
            for p in pred_lists[best_i]:
                earliest_start = max(earliest_start, sim_finish[p])
            
            candidate_times = [earliest_start]
            for finish_t, act_id in active:
                if finish_t > earliest_start:
                    candidate_times.append(finish_t)
            candidate_times.sort()
            
            feasible_start = earliest_start
            found = False
            for t_candidate in candidate_times:
                current_demand = np.zeros(n_res)
                for finish_t, act_id in active:
                    if sim_start[act_id] < t_candidate <= finish_t:
                        current_demand += d_matrix[act_id]
                
                if np.all(current_demand + d_matrix[best_i] <= capacities_arr):
                    feasible_start = t_candidate
                    found = True
                    break
            
            if not found:
                feasible_start = earliest_start
            
            sim_start[best_i] = feasible_start
            sim_finish[best_i] = feasible_start + durations[best_i]
            active.append((sim_finish[best_i], best_i))
            remaining.remove(best_i)
            scheduled.add(best_i)
            
            active = [(f, i) for f, i in active if f > feasible_start]
    
    # 6. Compute metrics for scoring
    
    # Normalize BL
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    bl_min = np.min(bottom_level)
    bl_range = bl_max - bl_min
    if bl_range > 0:
        bl_norm = (bottom_level - bl_min) / bl_range
    else:
        bl_norm = np.ones(n)
    
    # Soft-max resource contention penalty
    sm_norm = np.zeros(n)
    if d_matrix is not None and capacities_arr is not None and n_res > 0:
        ratios = d_matrix / capacities_arr
        temperature = 0.5
        ratios_clipped = np.clip(ratios, -50.0, 50.0)
        scaled = ratios_clipped / temperature
        max_scaled = np.max(scaled, axis=1, keepdims=True)
        max_scaled_safe = np.where(np.isfinite(max_scaled), max_scaled, 0.0)
        exp_scaled = np.exp(scaled - max_scaled_safe)
        log_sum_exp = max_scaled_safe + np.log(np.sum(exp_scaled, axis=1, keepdims=True))
        soft_max_penalty = log_sum_exp.squeeze(axis=1)
        sm_max = np.max(soft_max_penalty) if np.max(soft_max_penalty) > 0 else 1.0
        sm_min = np.min(soft_max_penalty)
        sm_range = sm_max - sm_min
        if sm_range > 0:
            sm_norm = (soft_max_penalty - sm_min) / sm_range
        else:
            sm_norm = np.zeros(n)
    
    # Duration normalization
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_min = np.min(durations)
    dur_range = dur_max - dur_min
    if dur_range > 0:
        dur_norm = (durations - dur_min) / dur_range
    else:
        dur_norm = np.ones(n)
    
    # Dynamic bottleneck indicator from simulation:
    # Activities that were delayed in simulation indicate resource contention
    # We want to prioritize activities that are both critical AND cause contention
    if d_matrix is not None and capacities_arr is not None and n_res > 0:
        delays = np.maximum(0, sim_start - es)
        del_max = np.max(delays) if np.max(delays) > 0 else 1.0
        if del_max > 0:
            del_norm = delays / del_max
        else:
            del_norm = np.zeros(n)
    else:
        del_norm = np.zeros(n)
    
    # 7. Combine scores
    # Primary: Bottom Level (criticality)
    # Secondary: Soft-max resource contention (bottleneck activities)
    # Tertiary: Duration (longer tasks first as tie-breaker)
    # Quaternary: Dynamic delay indicator (activities that caused delays)
    
    w_bl = 0.60
    w_sm = 0.20
    w_dur = 0.10
    w_del = 0.10
    
    scores = w_bl * bl_norm + w_sm * sm_norm + w_dur * dur_norm + w_del * del_norm
    
    # Normalize final scores to [0, 1]
    score_max = np.max(scores) if np.max(scores) > 0 else 1.0
    score_min = np.min(scores)
    score_range = score_max - score_min
    if score_range > 0:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones(n)

    return scores
# EVOLVE-BLOCK-END
