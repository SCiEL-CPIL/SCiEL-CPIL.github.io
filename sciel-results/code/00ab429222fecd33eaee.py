"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    """
    Compute priority scores using the Critical Path (Longest Path) heuristic.
    
    This approach assigns a priority to each task equal to the total processing
    time of the longest path starting from that task and ending at any terminal
    task. This identifies tasks that are part of the bottleneck of the assembly
    line. By prioritizing these tasks, we ensure that the constraints on the
    cycle time are addressed early, potentially reducing the total number of
    stations required.
    
    This differs from summing all successor durations (which measures total
    downstream work) by focusing on the maximum sequential delay introduced by
    a task.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([])
        
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
    
    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If cycle detected, fallback to times
    if len(topo_order) < n:
        return times.copy()
    
    # Compute longest path (Critical Path) in reverse topological order
    # longest_path[i] = times[i] + max(longest_path[s] for s in successors[i])
    longest_path = np.zeros(n, dtype=float)
    
    for node in reversed(topo_order):
        max_succ_path = 0.0
        for succ in successors[node]:
            if longest_path[succ] > max_succ_path:
                max_succ_path = longest_path[succ]
        longest_path[node] = times[node] + max_succ_path
        
    # The priority is the longest path length.
    # Higher priority means the task starts a longer chain, so it should be placed earlier
    # to avoid extending the cycle time of subsequent stations.
    return longest_path
# EVOLVE-BLOCK-END
