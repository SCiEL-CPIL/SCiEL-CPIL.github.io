"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque


def _compute_priorities(instance):
    """Compute a rich priority score for each task based on multiple heuristics."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Parse predecessors
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    elif isinstance(preds, dict):
        pred_list = [[] for _ in range(n)]
        for k, v in preds.items():
            idx = int(k)
            if 0 <= idx < n:
                pred_list[idx] = [int(p) for p in v]
        preds = pred_list
    elif isinstance(preds, list):
        preds = [[int(p) for p in row] for row in preds]
    
    # Build successors
    succs = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n:
                succs[p].append(i)
                in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
    
    # If topological sort didn't include all nodes (cycle), handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Compute longest path from each node to sink (downstream depth)
    # This represents the minimum time needed after this task if done immediately
    longest_downstream = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for u in reversed(topo_order):
        if succs[u]:
            longest_downstream[u] = max(
                times[v] + longest_downstream[v] for v in succs[u]
            )
        else:
            longest_downstream[u] = times[u]
    
    # Compute longest path from source to each node (upstream depth)
    longest_upstream = np.zeros(n, dtype=float)
    for u in topo_order:
        if preds[u]:
            longest_upstream[u] = max(
                times[p] + longest_upstream[p] for p in preds[u]
            )
    
    # Compute number of immediate successors (successor count)
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # Compute number of immediate predecessors
    pred_count = np.array([len(preds[i]) for i in range(n)], dtype=float)
    
    # Compute downstream total work (sum of times in downstream subtree)
    downstream_work = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        downstream_work[u] = times[u] + sum(downstream_work[v] for v in succs[u])
    
    # Compute upstream total work
    upstream_work = np.zeros(n, dtype=float)
    for u in topo_order:
        upstream_work[u] = times[u] + sum(upstream_work[p] for p in preds[u])
    
    # Minimum stations needed for downstream subtree (lower bound)
    # Based on total downstream work / cycle_time
    min_stations_downstream = np.ceil(downstream_work / cycle_time)
    min_stations_downstream = np.maximum(min_stations_downstream, 1)
    
    # Bottleneck density: time / min_stations_downstream
    bottleneck_density = times / min_stations_downstream
    
    # Combine multiple signals into a single priority score
    # Weighted combination:
    # 1. Processing time (longer tasks first) - weight 1.0
    # 2. Longest downstream path (critical path) - weight 0.8
    # 3. Successor count (more dependents = more urgent) - weight 0.3
    # 4. Bottleneck density - weight 0.5
    # 5. Downstream work (total remaining work) - weight 0.2
    
    # Normalize each component to [0, 1] range for better combination
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_time = normalize(times)
    norm_downstream = normalize(longest_downstream)
    norm_succ = normalize(succ_count)
    norm_bottleneck = normalize(bottleneck_density)
    norm_downwork = normalize(downstream_work)
    
    # Priority score
    priority = (
        1.0 * norm_time +
        0.8 * norm_downstream +
        0.3 * norm_succ +
        0.5 * norm_bottleneck +
        0.2 * norm_downwork
    )
    
    return priority


def priority(instance):
    """Score each assembly task; higher priority tasks are placed first."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    try:
        scores = _compute_priorities(instance)
        
        # Check deadline
        if time.time() > deadline:
            # Fallback to simple time-based priority
            return np.asarray(instance["times"], dtype=float)
        
        return scores
        
    except Exception:
        # Fallback to simple time-based priority
        return np.asarray(instance["times"], dtype=float)
# EVOLVE-BLOCK-END
