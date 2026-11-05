"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 130.0
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: Resource-weighted criticality combining Bottom Level,
    Total Float, and empirical bottleneck detection from a greedy simulation.
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
    # BL[i] = duration[i] + max(BL[j] for j in successors[i])
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            max_succ_bl = np.max([bottom_level[v] for v in successors[u]])
        else:
            max_succ_bl = 0.0
        bottom_level[u] = durations[u] + max_succ_bl

    # 3. Compute Resource Pressure
    # Sum of (demand / capacity) for each resource
    rp = np.zeros(n)
    n_res = 0
    d_matrix = None
    capacities_arr = None
    if demands is not None and capacities is not None:
        capacities_arr = np.asarray(capacities, dtype=float)
        if len(capacities_arr) == 0:
            capacities_arr = np.array([1.0])
        else:
            capacities_arr = np.where(capacities_arr > 0, capacities_arr, 1.0)
            
        # Process demands
        n_res = len(capacities_arr)
        # Initialize demand matrix
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
                        # Scalar?
                        if n_res > 0:
                            d_matrix[i, 0] = float(d_i)
        else:
            # Try converting whole thing to array
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

        # Calculate pressure: sum(d / c)
        # d_matrix shape (n, n_res), capacities shape (n_res,)
        ratios = d_matrix / capacities_arr
        rp = np.sum(ratios, axis=1)

    # 4. Normalize Metrics
    # Bottom Level
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    bl_min = np.min(bottom_level)
    bl_range = bl_max - bl_min
    if bl_range > 0:
        bl_norm = (bottom_level - bl_min) / bl_range
    else:
        bl_norm = np.ones(n)

    # Resource Pressure
    rp_max = np.max(rp) if np.max(rp) > 0 else 1.0
    rp_min = np.min(rp)
    rp_range = rp_max - rp_min
    if rp_range > 0:
        rp_norm = (rp - rp_min) / rp_range
    else:
        rp_norm = np.ones(n)

    # Duration
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_min = np.min(durations)
    dur_range = dur_max - dur_min
    if dur_range > 0:
        dur_norm = (durations - dur_min) / dur_range
    else:
        dur_norm = np.ones(n)

    # 5. Simulate a resource-feasible greedy schedule to identify actual bottlenecks
    # This gives us empirical information about which activities cause resource contention
    bottleneck_score = np.zeros(n)
    bn_norm = np.zeros(n)
    if demands is not None and capacities is not None and n_res > 0:
        # Run a simple SGS simulation using bottom-level priority
        sim_priority = bottom_level.copy()
        remaining = set(range(n))
        start_times = np.full(n, -1.0)
        finish_times = np.full(n, -1.0)
        time_now = 0.0
        scheduled = 0
        max_sim_time = 10000.0  # safety bound
        
        # Precompute active intervals per resource for efficiency
        while scheduled < n and time_now < max_sim_time:
            # Find eligible activities (all predecessors completed)
            eligible = []
            for i in remaining:
                if all(start_times[p] >= 0 and finish_times[p] <= time_now for p in pred_lists[i]):
                    eligible.append(i)
            
            if not eligible:
                # Advance time to next event
                future_starts = [t for t in finish_times if t > time_now]
                if future_starts:
                    time_now = min(future_starts)
                else:
                    break
                continue
            
            # Select highest priority eligible activity
            best = max(eligible, key=lambda x: sim_priority[x])
            
            # Check resource feasibility using vectorized approach
            feasible = True
            for r in range(n_res):
                # Count current usage of resource r
                current_usage = 0.0
                for j in range(n):
                    if start_times[j] >= 0 and start_times[j] <= time_now < finish_times[j]:
                        if d_matrix[j, r] > 0:
                            current_usage += d_matrix[j, r]
                if current_usage + d_matrix[best, r] > capacities_arr[r] + 1e-9:
                    feasible = False
                    break
            
            if feasible:
                start_times[best] = time_now
                finish_times[best] = time_now + durations[best]
                remaining.discard(best)
                scheduled += 1
            else:
                # Cannot start now; advance time
                future_starts = [t for t in finish_times if t > time_now and t >= 0]
                if future_starts:
                    time_now = min(future_starts)
                else:
                    # Force schedule to avoid infinite loop
                    start_times[best] = time_now
                    finish_times[best] = time_now + durations[best]
                    remaining.discard(best)
                    scheduled += 1
        
        # Identify bottleneck activities: those that were delayed due to resource contention
        # or that consume high fractions of resources during their execution
        bottleneck_score = np.zeros(n)
        for i in range(n):
            if start_times[i] < 0:
                continue
            for r in range(n_res):
                if d_matrix[i, r] == 0:
                    continue
                # Count how many other activities overlap with i on resource r
                overlap_count = 0
                for j in range(n):
                    if j == i:
                        continue
                    if start_times[j] < 0:
                        continue
                    if start_times[i] < finish_times[j] and start_times[j] < finish_times[i]:
                        if d_matrix[j, r] > 0:
                            overlap_count += 1
                if overlap_count > 0:
                    fraction = d_matrix[i, r] / (capacities_arr[r] + 1e-9)
                    bottleneck_score[i] += fraction * min(overlap_count, 3.0)
        
        # Normalize bottleneck score
        bn_max = np.max(bottleneck_score) if np.max(bottleneck_score) > 0 else 1.0
        bn_min = np.min(bottleneck_score)
        bn_range = bn_max - bn_min
        if bn_range > 0:
            bn_norm = (bottleneck_score - bn_min) / bn_range
        else:
            bn_norm = np.zeros(n)

    # Compute Total Float (TF) for each activity
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
    
    tf_max = np.max(tf) if np.max(tf) > 0 else 1.0
    tf_min = np.min(tf)
    tf_range = tf_max - tf_min
    if tf_range > 0:
        tf_norm = (tf - tf_min) / tf_range
    else:
        tf_norm = np.zeros(n)
        
    # Combine scores with dynamic bottleneck awareness
    # Weighted combination: criticality (BL) + duration - resource pressure - total float
    # + bonus for activities on critical path that also have high bottleneck score
    w_bl = 0.50
    w_dur = 0.05
    w_rp = 0.10
    w_tf = 0.20
    w_bn = 0.15  # weight for bottleneck-aware bonus
    
    # Dynamic weighting: boost BL weight for activities with high bottleneck score
    # This creates a "resource-weighted criticality" effect
    bl_weighted = bl_norm * (1.0 + 0.3 * bn_norm)
    
    scores = w_bl * bl_weighted + w_dur * dur_norm - w_rp * rp_norm - w_tf * tf_norm + w_bn * bn_norm * bl_norm

    # Ensure scores are in a reasonable range
    score_max = np.max(scores) if np.max(scores) > 0 else 1.0
    score_min = np.min(scores)
    score_range = score_max - score_min
    if score_range > 0:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones(n)

    return scores
# EVOLVE-BLOCK-END
