"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores for each assembly task using an enhanced Critical Path Density metric.
    
    The score combines:
    1. Critical Path Position: the longest downstream path length (absolute criticality).
    2. Downstream Workload: sum of all immediate successor path lengths (total downstream work).
    3. Branching Factor: count of immediate successors (fanout).
    4. Task Duration: as a tiebreaker for near-equal scores.
    
    Tasks with high downstream criticality and workload are prioritized to minimize stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor lists from predecessors
    successors = [[] for _ in range(n)]
    for i in range(n):
        if i in predecessors:
            for pred in predecessors[i]:
                if pred is not None:
                    successors[pred].append(i)
    
    # Compute in-degrees for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        if i in predecessors:
            for pred in predecessors[i]:
                if pred is not None:
                    in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
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
    
    # If topo_order doesn't include all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes in arbitrary order
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)
    
    # Compute longest downstream path (including self) in reverse topological order
    longest_downstream = np.zeros(n, dtype=float)
    # Also compute sum of all successor path lengths (downstream workload)
    downstream_workload = np.zeros(n, dtype=float)
    
    for node in reversed(topo_order):
        succs = successors[node]
        if not succs:
            longest_downstream[node] = times[node]
            downstream_workload[node] = 0.0
        else:
            max_succ_path = 0.0
            total_succ_work = 0.0
            for s in succs:
                if longest_downstream[s] > max_succ_path:
                    max_succ_path = longest_downstream[s]
                total_succ_work += longest_downstream[s]
            longest_downstream[node] = times[node] + max_succ_path
            downstream_workload[node] = total_succ_work
    
    # Compute immediate successor count
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Normalize each component to [0, 1] for weighted combination
    max_ld = longest_downstream.max() if n > 0 else 1.0
    max_dw = downstream_workload.max() if n > 0 else 1.0
    max_sc = succ_count.max() if n > 0 else 1.0
    max_t = times.max() if n > 0 else 1.0
    
    if max_ld > 0:
        norm_ld = longest_downstream / max_ld
    else:
        norm_ld = np.zeros(n, dtype=float)
    
    if max_dw > 0:
        norm_dw = downstream_workload / max_dw
    else:
        norm_dw = np.zeros(n, dtype=float)
    
    if max_sc > 0:
        norm_sc = succ_count / max_sc
    else:
        norm_sc = np.zeros(n, dtype=float)
    
    if max_t > 0:
        norm_t = times / max_t
    else:
        norm_t = np.zeros(n, dtype=float)
    
    # Weighted combination:
    # - Longest downstream path is the primary criticality indicator (weight 0.5)
    # - Downstream workload captures total branching work (weight 0.3)
    # - Successor count adds branching awareness (weight 0.1)
    # - Task duration as a small tiebreaker (weight 0.1)
    priority_scores = 0.5 * norm_ld + 0.3 * norm_dw + 0.1 * norm_sc + 0.1 * norm_t
    
    return priority_scores
# EVOLVE-BLOCK-END
