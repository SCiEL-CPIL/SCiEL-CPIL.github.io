"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def _compute_downstream_times(times, successors):
    """Compute total downstream time (sum of times of all transitive successors)."""
    n = len(times)
    # Topological order (Kahn's algorithm)
    in_degree = [0] * n
    for i in range(n):
        for s in successors[i]:
            in_degree[s] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If graph has cycles, fall back to simple ordering
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Reverse topological order to compute downstream sums
    downstream_sum = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        s = sum(downstream_sum[succ] for succ in successors[node])
        downstream_sum[node] = times[node] + s
    
    # downstream_time = total downstream minus own time
    downstream_only = downstream_sum - times
    
    return downstream_only, topo_order


def _compute_rank_from_end(successors, n):
    """Compute the longest path length from each node to any sink (rank from end)."""
    in_degree = [0] * n
    for i in range(n):
        for s in successors[i]:
            in_degree[s] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    if len(topo_order) < n:
        return np.zeros(n, dtype=int)
    
    # Rank from end: longest path to a sink
    rank_end = np.zeros(n, dtype=int)
    for node in reversed(topo_order):
        if successors[node]:
            rank_end[node] = 1 + max(rank_end[s] for s in successors[node])
        else:
            rank_end[node] = 0
    
    return rank_end


def _compute_weighted_positional_rank(times, successors, n):
    """Compute a weighted positional rank that considers both time and position."""
    in_degree = [0] * n
    for i in range(n):
        for s in successors[i]:
            in_degree[s] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    if len(topo_order) < n:
        return np.zeros(n, dtype=float)
    
    # For each node, compute the max weighted path to any sink
    # weight = time of node
    max_path_to_sink = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if successors[node]:
            max_path_to_sink[node] = times[node] + max(max_path_to_sink[s] for s in successors[node])
        else:
            max_path_to_sink[node] = times[node]
    
    return max_path_to_sink


def priority(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    cycle_time = instance.get("cycle_time", 40.0)

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)

    # Compute various metrics
    downstream_only, topo_order = _compute_downstream_times(times, successors)
    rank_end = _compute_rank_from_end(successors, n)
    weighted_pos_rank = _compute_weighted_positional_rank(times, successors, n)
    
    # Count direct successors
    direct_succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Count direct predecessors
    direct_pred_count = np.array([len(preds[i]) for i in range(n)], dtype=float)
    
    # Compute "slack" relative to cycle time - tasks that are close to cycle time
    # should be prioritized as they constrain station capacity
    slack_ratio = times / max(cycle_time, 1.0)
    
    # Normalize all metrics
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_downstream = normalize(downstream_only)
    norm_rank_end = normalize(rank_end.astype(float))
    norm_weighted = normalize(weighted_pos_rank)
    norm_direct_succ = normalize(direct_succ_count)
    norm_slack = normalize(slack_ratio)
    norm_time = normalize(times)
    
    # Combined priority: 
    # - High downstream time means this task unlocks many subsequent tasks
    # - High rank from end means this task is early in the critical path
    # - High weighted positional rank means this task is on a long path
    # - High direct successor count means high fan-out
    # - High slack ratio means this task uses a large fraction of cycle time
    # - High time means this task is large
    
    # Weights tuned to emphasize bottleneck tasks
    priority_scores = (
        0.35 * norm_downstream +
        0.25 * norm_rank_end +
        0.15 * norm_weighted +
        0.10 * norm_direct_succ +
        0.10 * norm_slack +
        0.05 * norm_time
    )
    
    # Deterministic tie-breaker based on topological position
    # Earlier in topo order gets slight boost
    topo_pos = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        topo_pos[node] = idx
    norm_topo_pos = normalize(topo_pos)
    
    priority_scores = priority_scores - 0.01 * norm_topo_pos  # Earlier topo pos slightly preferred
    
    # Tiny deterministic tie-breaker
    priority_scores = priority_scores + np.arange(n) * 1e-12

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return times  # Fallback to simple time-based

    return priority_scores
# EVOLVE-BLOCK-END
