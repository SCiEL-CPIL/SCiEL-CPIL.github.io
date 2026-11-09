"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (n, n_res)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (n_res,)
    predecessors = instance["predecessors"]  # list of lists
    
    n = len(durations)
    n_res = len(capacities)
    
    # Build adjacency: successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Topological order via Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        # Pick from queue (stable)
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't include all nodes, append remaining (shouldn't happen in valid instances)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Forward pass: earliest start (ES) and earliest finish (EF)
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in topo_order:
        es[u] = 0
        for p in predecessors[u]:
            es[u] = max(es[u], ef[p])
        ef[u] = es[u] + durations[u]
    
    # Backward pass: latest start (LS) and latest finish (LF)
    project_end = max(ef) if n > 0 else 0
    lf = np.full(n, project_end)
    ls = np.full(n, project_end)
    # Process in reverse topological order
    for u in reversed(topo_order):
        lf[u] = project_end
        for v in successors[u]:
            lf[u] = min(lf[u], ls[v])
        ls[u] = lf[u] - durations[u]
    
    # Slack
    slack = ls - es
    
    # Resource utilization: for each activity, how much of each resource it uses relative to capacity
    # Higher demand relative to capacity means more constraining
    # Compute a "resource pressure" score for each activity
    # If demand is high relative to capacity, it's more likely to cause resource conflicts
    
    # Normalize demands by capacity to get utilization ratio
    # demands: (n, n_res), capacities: (n_res,)
    util = demands / capacities[np.newaxis, :]  # (n, n_res)
    
    # Max utilization per activity (worst-case resource constraint)
    max_util = np.max(util, axis=1) if n_res > 0 else np.zeros(n)
    
    # Mean utilization per activity
    mean_util = np.mean(util, axis=1) if n_res > 0 else np.zeros(n)
    
    # Combine: prioritize activities that:
    # 1. Have low slack (critical path)
    # 2. Have high resource utilization (constraining)
    # 3. Have long duration
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        if len(arr) == 0:
            return arr
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    # Low slack is good -> invert slack
    slack_inv = normalize(-slack)
    dur_norm = normalize(durations)
    max_util_norm = normalize(max_util)
    mean_util_norm = normalize(mean_util)
    
    # Weighted combination
    # Emphasize critical path (low slack) and resource constraints
    score = (0.4 * slack_inv + 0.3 * max_util_norm + 0.2 * dur_norm + 0.1 * mean_util_norm)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Safety: return a simple fallback
        return np.asarray(durations, dtype=float)
    
    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
