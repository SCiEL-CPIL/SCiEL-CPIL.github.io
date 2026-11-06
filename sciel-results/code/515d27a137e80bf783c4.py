"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    """
    Compute priority scores using Critical Path Method (CPM) and Descendant Count.
    
    This approach differs from Positional Weight (PW) by focusing on the
    longest bottleneck path (Critical Path) rather than the sum of all
    downstream work. This helps ensure that long chains of tasks are
    completed within fewer stations, reducing the overall line length.
    
    Score = Critical_Path_Length + alpha * Normalized_Descendant_Count
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
    
    # Compute in-degree
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
    
    if len(topo_order) < n:
        # Cycle detected, fallback to times
        return times.copy()
    
    # 1. Compute Critical Path Length (CPL)
    # CPL[i] = times[i] + max(CPL[succ])
    cpl = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ_cpl = 0.0
        for succ in successors[node]:
            if cpl[succ] > max_succ_cpl:
                max_succ_cpl = cpl[succ]
        cpl[node] = times[node] + max_succ_cpl
        
    # 2. Compute Total Descendant Count (TDC)
    # TDC[i] = number of all tasks that i precedes (transitively)
    tdc = np.zeros(n, dtype=int)
    for node in reversed(topo_order):
        count = 0
        for succ in successors[node]:
            count += 1 + tdc[succ]
        tdc[node] = count
        
    # Normalize TDC
    max_tdc = np.max(tdc) if n > 0 else 1
    if max_tdc == 0:
        norm_tdc = np.zeros(n, dtype=float)
    else:
        norm_tdc = tdc / max_tdc
        
    # Combine scores
    # CPL is the primary driver (bottleneck focus)
    # Norm TDC is a secondary driver (branching focus)
    alpha = 0.1
    scores = cpl + alpha * norm_tdc
    
    return scores
# EVOLVE-BLOCK-END
