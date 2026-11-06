"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute priority scores for each activity using a composite metric:
    - Backward critical path length (longest path to project end)
    - Resource scarcity (demand/capacity ratio summed across resources)
    - Resource bottleneck centrality: downstream resource pressure gated by this activity
    - Downstream fan-out weighted by downstream criticality
    
    Higher priority = scheduled earlier in serial SGS.
    """
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse predecessors
    predecessors = instance.get("predecessors", [])
    # Build successor list
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Topological sort
    in_degree = pred_count[:]
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topo_order doesn't have all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        return durations.copy()
    
    # Backward pass: dist_to_end[i] = durations[i] + max(dist_to_end[succ] for succ in successors[i])
    dist_to_end = np.zeros(n)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        max_succ = 0.0
        for succ in successors[node]:
            if dist_to_end[succ] > max_succ:
                max_succ = dist_to_end[succ]
        dist_to_end[node] = durations[node] + max_succ
    
    # Resource scarcity
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_score = np.zeros(n)
    num_resources = 0
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 1:
            # Single resource or per-activity demand
            if capacities.ndim == 0:
                cap = float(capacities)
                if cap > 0:
                    resource_score = demands / cap
                else:
                    resource_score = np.zeros(n)
                num_resources = 1
            else:
                cap = float(np.max(capacities)) if len(capacities) > 0 else 1.0
                if cap > 0:
                    resource_score = demands / cap
                else:
                    resource_score = np.zeros(n)
                num_resources = 1
        else:
            # demands is (n, R), capacities is (R,)
            safe_caps = np.where(capacities > 0, capacities, 1.0)
            ratio = demands / safe_caps[np.newaxis, :]
            resource_score = np.sum(ratio, axis=1)
            num_resources = ratio.shape[1]
    
    # Resource bottleneck centrality:
    # For each activity, compute the total "resource pressure" of all downstream activities.
    # This captures how much resource-constrained work depends on this activity completing.
    # bottleneck_central[node] = sum over all downstream d of resource_score[d] * (1/num_predecessors_on_path)
    # Simplified: accumulate resource_score of descendants in reverse topo order
    bottleneck_central = np.zeros(n)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        total = 0.0
        for succ in successors[node]:
            total += resource_score[succ] + bottleneck_central[succ]
        bottleneck_central[node] = total
    
    # Downstream criticality mass
    descendant_crit = np.zeros(n)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        total = 0.0
        for succ in successors[node]:
            total += dist_to_end[succ] + descendant_crit[succ]
        descendant_crit[node] = total
    
    # Check time budget after expensive computations
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # Return what we have with basic normalization
        def normalize(arr):
            mn = np.min(arr)
            mx = np.max(arr)
            if mx - mn < 1e-12:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)
        priority_scores = 0.5 * normalize(dist_to_end) + 0.3 * normalize(resource_score) + 0.2 * normalize(descendant_crit)
        return priority_scores
    
    # Normalize components to [0, 1] for combining
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_dist = normalize(dist_to_end)
    norm_res = normalize(resource_score)
    norm_desc = normalize(descendant_crit)
    norm_bottleneck = normalize(bottleneck_central)
    norm_dur = normalize(durations)
    
    # Combine: weighted sum emphasizing critical path, resource bottleneck, and local scarcity
    # Resource bottleneck centrality captures which activities gate the most resource-constrained downstream work
    priority_scores = 0.35 * norm_dist + 0.25 * norm_bottleneck + 0.2 * norm_res + 0.15 * norm_desc + 0.05 * norm_dur
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return priority_scores
    
    return priority_scores
# EVOLVE-BLOCK-END
