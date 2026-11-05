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
    
    Heuristic: CPM bottom level + dynamic resource-contention priority.
    For each activity, compute the max resource utilization ratio among
    all activities sharing any resource, combined with longest path to end.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        durations = np.asarray(instance["durations"], dtype=float)
        if len(durations) == 0:
            return np.array([])
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
            pred_lists.append([x for x in p if x is not None and 0 <= x < n])
        else:
            pred_lists.append([p] if 0 <= p < n else [])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
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

    # 2. Compute Bottom Level (longest path from node to end)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            max_succ = max(bottom_level[v] + durations[v] for v in successors[u])
        else:
            max_succ = 0.0
        bottom_level[u] = durations[u] + max_succ

    # 3. Compute CPM Total Float
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        if pred_lists[u]:
            es[u] = max(ef[p] for p in pred_lists[u])
        ef[u] = es[u] + durations[u]
        
    cp_length = np.max(ef) if n > 0 else 0.0
    
    ls = np.zeros(n)
    lf = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            lf[u] = min(ls[v] for v in successors[u])
        else:
            lf[u] = cp_length
        ls[u] = lf[u] - durations[u]
        
    total_float = ls - es

    # 4. Process demands and capacities
    d_matrix = None
    capacities_arr = None
    n_res = 0
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
                            if 0 <= r < n_res:
                                d_matrix[i, r] = float(val)
                    elif isinstance(d_i, (list, tuple, np.ndarray)):
                        d_arr = np.asarray(d_i, dtype=float)
                        if d_arr.ndim == 1:
                            min_len = min(len(d_arr), n_res)
                            d_matrix[i, :min_len] = d_arr[:min_len]
                    else:
                        try:
                            if n_res > 0:
                                d_matrix[i, 0] = float(d_i)
                        except:
                            pass
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

    # 5. Dynamic Resource-Contention Priority
    # For each activity i, find the maximum (demand/capacity) ratio among
    # all activities j that share at least one resource with i.
    # This identifies activities that are in the most contested resource pools.
    contention_score = np.zeros(n)
    if d_matrix is not None and capacities_arr is not None and n_res > 0:
        # For each resource r, find the max ratio among activities using r
        # Then for each activity i, contention = max over resources r that i uses
        #   of (max ratio among all activities using r)
        # This captures "how contested is the most contested resource I use"
        
        # Compute ratio matrix
        ratios = d_matrix / capacities_arr  # shape (n, n_res)
        
        # For each resource r, the max ratio among activities using it
        # (activities with 0 demand don't use the resource)
        # We want: for activity i, max over r where d_matrix[i,r] > 0 of
        #   max_j (ratios[j, r]) where d_matrix[j, r] > 0
        
        # Efficient approach: for each resource, compute the max ratio
        # then for each activity, take max over resources it uses
        max_ratio_per_resource = np.zeros(n_res)
        for r in range(n_res):
            using_r = ratios[:, r] > 0
            if np.any(using_r):
                max_ratio_per_resource[r] = np.max(ratios[using_r, r])
        
        # For each activity, contention = max over resources it uses of max_ratio_per_resource
        # Use broadcasting: for each i, max over r where d_matrix[i,r]>0
        # Set non-used resources to -inf for each activity
        mask = d_matrix > 0  # (n, n_res)
        # Create a matrix where unused resources get -inf
        contention_matrix = np.where(mask, max_ratio_per_resource[np.newaxis, :], -np.inf)
        # For activities using no resources, contention = 0
        no_resource = ~np.any(mask, axis=1)
        contention_matrix[no_resource, :] = 0
        contention_score = np.max(contention_matrix, axis=1)
        
        # Also compute the activity's own max ratio (how much it stresses resources)
        own_max_ratio = np.max(ratios, axis=1)
        
        # Combined: activities in contested pools AND that themselves stress resources
        # are highest priority
        # Normalize
        cs_max = np.max(contention_score) if np.max(contention_score) > 0 else 1.0
        omr_max = np.max(own_max_ratio) if np.max(own_max_ratio) > 0 else 1.0
        cs_norm = contention_score / cs_max
        omr_norm = own_max_ratio / omr_max
    else:
        cs_norm = np.zeros(n)
        omr_norm = np.zeros(n)

    # 6. Normalize all metrics
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    bl_min = np.min(bottom_level)
    bl_range = bl_max - bl_min
    if bl_range > 0:
        bl_norm = (bottom_level - bl_min) / bl_range
    else:
        bl_norm = np.ones(n)

    tf_max = np.max(total_float) if np.max(total_float) > 0 else 1.0
    tf_min = np.min(total_float)
    tf_range = tf_max - tf_min
    if tf_range > 0:
        tf_norm = (total_float - tf_min) / tf_range
    else:
        tf_norm = np.zeros(n)
    # Invert: low float = high priority
    tf_inv = 1.0 - tf_norm

    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_min = np.min(durations)
    dur_range = dur_max - dur_min
    if dur_range > 0:
        dur_norm = (durations - dur_min) / dur_range
    else:
        dur_norm = np.ones(n)

    # 7. Combine scores
    # Primary: bottom level (critical path proximity)
    # Secondary: dynamic resource contention (contested resource pools)
    # Tertiary: own resource pressure
    # Quaternary: low total float
    # Quinary: duration (longer tasks benefit from early start)
    
    w_bl = 0.45
    w_cs = 0.20  # contention score
    w_omr = 0.10  # own max ratio
    w_tf = 0.15  # inverted total float
    w_dur = 0.10  # duration
    
    scores = (w_bl * bl_norm + 
              w_cs * cs_norm + 
              w_omr * omr_norm + 
              w_tf * tf_inv + 
              w_dur * dur_norm)

    # Final normalization
    score_max = np.max(scores) if np.max(scores) > 0 else 1.0
    score_min = np.min(scores)
    score_range = score_max - score_min
    if score_range > 0:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones(n)

    return scores
# EVOLVE-BLOCK-END
