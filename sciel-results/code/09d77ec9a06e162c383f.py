import numpy as np
from collections import deque

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    
    # Handle case where n is 0
    if n == 0:
        return np.array([])

    # Build successors list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort (Kahn's Algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If there's a cycle, topo_order will be incomplete. 
    # RCPSP instances are DAGs, so this shouldn't happen.
    # But just in case, if len(topo_order) < n, we might have issues.
    # For safety, if incomplete, we fall back to LDF?
    if len(topo_order) < n:
         return durations

    # Calculate Longest Path to End (LD)
    LD = np.zeros(n)
    # Process in reverse topological order
    for u in reversed(topo_order):
        max_succ_LD = 0.0
        for v in successors[u]:
            if LD[v] > max_succ_LD:
                max_succ_LD = LD[v]
        LD[u] = durations[u] + max_succ_LD
        
    # Add a tiny tie-breaker: prefer longer durations if LD is equal
    # This mimics LDF for ties, which is known to be good.
    return LD + durations * 1e-9
