import numpy as np
from collections import defaultdict, deque
import time

def priority(instance):
    """
    Compute priority scores for assembly tasks.
    Higher priority = more important to assign earlier.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance.get("cycle_time", None)
    
    n = len(times)
    if n == 0:
        return np.array([], dtype=float)
    
    # Build graph structures
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        if isinstance(predecessors, dict):
            preds = predecessors.get(i, [])
        else:
            preds = predecessors[i]
        
        for p in preds:
            succ[p].append(i)
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
    
    if len(topo_order) < n:
        # Cycle detected - fallback to times
        return times
    
    # Compute rank: longest path from node to any sink (inclusive of node time)
    rank = [0.0] * n
    for u in reversed(topo_order):
        max_succ_rank = 0.0
        for v in succ[u]:
            if rank[v] > max_succ_rank:
                max_succ_rank = rank[v]
        rank[u] = times[u] + max_succ_rank
    
    # Compute downstream workload sum: sum of times of all successors (transitive)
    # Process in reverse topological order
    down_work = [0.0] * n
    for u in reversed(topo_order):
        sum_succ = 0.0
        for v in succ[u]:
            sum_succ += times[v] + down_work[v]
        down_work[u] = sum_succ
    
    # Compute out-degree
    out_degree = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Compute in-degree (number of predecessors)
    in_deg = np.array(in_degree, dtype=float)
    
    # Combine features into a priority score
    # Strategy: prioritize tasks that are on long critical paths AND have high downstream impact
    # High rank means the task is part of a long chain
    # High down_work means completing this task unlocks a lot of work
    # High out_degree means this task directly enables many successors
    
    # Normalize each component to [0, 1] range
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_rank = normalize(rank)
    norm_down = normalize(down_work)
    norm_outdeg = normalize(out_degree)
    norm_time = normalize(times)
    
    # Weighted combination
    # Emphasize critical path position and downstream impact
    score = (1.2 * norm_rank + 
             0.8 * norm_down + 
             0.5 * norm_outdeg + 
             0.3 * norm_time)
    
    # Add small tie-breaking by task duration to ensure deterministic ordering
    eps = 1e-9
    score = score + eps * norm_time
    
    return score
