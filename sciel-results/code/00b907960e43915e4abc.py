"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores for each task using an improved 'bottleneck density' heuristic.
    
    The previous approach used a ratio that inadvertently deprioritized critical
    tasks with large downstream subtrees. This version reinterprets the bottleneck
    density to prioritize tasks that are both long (high processing time) and
    critical (high downstream station requirement), effectively approximating
    the product of task time and critical path length. This aligns with standard
    Assembly Line Balancing heuristics (like Scholl's) that favor high critical
    path tasks to unlock successors early.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Build adjacency list: predecessors[i] is a list of task indices that must come before i
    preds = instance.get("predecessors", [None] * n)
    
    # Build successor list: succ[i] = list of tasks that depend on i
    succ = [[] for _ in range(n)]
    for i in range(n):
        if preds[i] is not None:
            for p in preds[i]:
                succ[p].append(i)
    
    # Compute in-degrees for topological sort
    in_degree = [0] * n
    for i in range(n):
        if preds[i] is not None:
            for p in preds[i]:
                in_degree[i] += 1
    
    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)
    
    # Compute longest path from each node to a leaf (downstream critical path)
    # longest_down[node] represents the total time from 'node' to the end of the line
    # along the longest path.
    longest_down = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if succ[node]:
            longest_down[node] = times[node] + max(longest_down[s] for s in succ[node])
        else:
            longest_down[node] = times[node]
    
    # Estimate the minimum number of stations required to complete the downstream path.
    # Using the critical path length is a tighter lower bound on the number of stations
    # needed for the critical chain than the total subtree sum.
    # We use this as the "Min Remaining Stations" component of the density.
    min_remaining_stations = np.maximum(1.0, np.ceil(longest_down / cycle_time))
    
    # The "Bottleneck Density" is redefined here to prioritize tasks that have
    # high processing time AND high downstream criticality.
    # Instead of Time / Stations (which favored leaves), we use Time * Stations
    # (or equivalently, Time / (1/Stations)). This rewards long tasks that are
    # part of long critical chains.
    # To avoid division by zero issues and keep it a "ratio" conceptually,
    # we can view this as Time / (1/MinStations).
    # However, simple multiplication is numerically stable and effective.
    
    # We add a small bonus for immediate successors to break ties in favor of
    # tasks that unlock more immediate work.
    succ_counts = np.array([len(s) for s in succ], dtype=float)
    
    # Normalize successor counts to [0, 1] for a consistent bonus scale
    max_succ = succ_counts.max() if n > 0 else 1.0
    if max_succ > 0:
        succ_bonus = succ_counts / max_succ
    else:
        succ_bonus = np.zeros(n)
    
    # Final Priority: (Time * MinStations) + Small Successor Bonus
    # This ensures tasks with high critical path involvement are prioritized.
    priority_scores = (times * min_remaining_stations) + 0.1 * succ_bonus
    
    return priority_scores
# EVOLVE-BLOCK-END
