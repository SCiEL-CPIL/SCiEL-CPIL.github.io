import numpy as np
import time
from collections import deque


def priority(instance):
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Build predecessor lists
    predecessors = instance.get("predecessors", [])
    # predecessors[i] is a list of activity indices that must complete before i starts
    
    # Build successors and in-degree
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Forward pass: earliest start time (EST) for each activity
    est = np.zeros(n, dtype=float)
    for node in topo_order:
        max_pred_end = 0.0
        preds = predecessors[node] if node < len(predecessors) else []
        for p in preds:
            if est[p] + durations[p] > max_pred_end:
                max_pred_end = est[p] + durations[p]
        est[node] = max_pred_end
    
    # Backward pass: latest finish time (LFT) for each activity
    # First compute project makespan (unconstrained CPM)
    makespan = max(est[i] + durations[i] for i in range(n)) if n > 0 else 0.0
    
    lft = np.full(n, makespan, dtype=float)
    for node in reversed(topo_order):
        max_succ_lft = makespan
        for s in successors[node]:
            if lft[s] < max_succ_lft:
                max_succ_lft = lft[s]
        lft[node] = max_succ_lft
    
    # Total float (slack) = LFT - EST - duration
    # Low float = critical path activity, should be scheduled first
    total_float = lft - est - durations
    
    # Resource demand factor: use max relative demand (binding resource)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_score = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 2:
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            rel_demand = demands / safe_cap
            # Max relative demand captures the most constrained resource
            resource_score = np.max(rel_demand, axis=1)
        else:
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            resource_score = demands / safe_cap
    
    # Check deadline
    if time.time() - t_start > DEADLINE:
        # Fallback: just use inverse float
        if total_float.max() > total_float.min():
            float_norm = (total_float.max() - total_float) / (total_float.max() - total_float.min())
        else:
            float_norm = np.ones(n)
        return float_score
    
    # Normalize total float: invert so that low float (critical) gets high priority
    tf_min, tf_max = total_float.min(), total_float.max()
    if tf_max > tf_min:
        float_norm = (tf_max - total_float) / (tf_max - tf_min)  # 1 = most critical
    else:
        float_norm = np.ones(n)
    
    # Normalize resource score
    rs_min, rs_max = resource_score.min(), resource_score.max()
    if rs_max > rs_min:
        rs_norm = (resource_score - rs_min) / (rs_max - rs_min)
    else:
        rs_norm = np.zeros(n)
    
    # Normalize duration as a small tie-breaker
    d_min, d_max = durations.min(), durations.max()
    if d_max > d_min:
        dur_norm = (durations - d_min) / (d_max - d_min)
    else:
        dur_norm = np.zeros(n)
    
    # Weighted combination: float is dominant, resource demand secondary, duration as tie-breaker
    priority_scores = 0.80 * float_norm + 0.15 * rs_norm + 0.05 * dur_norm
    
    return priority_scores
