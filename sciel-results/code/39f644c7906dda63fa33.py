"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute positional weight: time[i] + sum of positional weights of successors
    # Use topological order (reverse topological: process sinks first)
    # Compute in-degree for topological sort
    in_degree = [len(predecessors[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If topological sort didn't include all nodes (cycle), fallback
    if len(topo_order) < n:
        # Fallback: just use times
        return times.copy()
    
    # Reverse topological order: process from sinks to sources
    rev_topo = list(reversed(topo_order))
    
    # Compute positional weight
    pos_weight = np.zeros(n, dtype=float)
    for i in rev_topo:
        pos_weight[i] = times[i]
        for s in successors[i]:
            pos_weight[i] += pos_weight[s]
    
    # Count immediate successors
    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    max_succ = max(num_succ.max(), 1.0)
    
    # Bottleneck measure: time[i] / (pos_weight[i] - time[i] + 1) 
    # This is high when the task's own time dominates its downstream path
    downstream_weight = np.maximum(pos_weight - times, 1e-9)
    bottleneck = times / downstream_weight
    
    # Composite priority: primarily positional weight, with small boosts
    # for successor count and bottleneck-ness
    priority_score = pos_weight + 0.1 * (num_succ / max_succ) + 0.05 * bottleneck
    
    return priority_score
# EVOLVE-BLOCK-END
