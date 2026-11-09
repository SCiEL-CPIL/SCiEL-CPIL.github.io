import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor graph
    # predecessors[i] is a list of indices of tasks that must come before i
    successors = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological sort (Kahn's algorithm) to get order
    # We want reverse topological order to compute PW
    # PW depends on successors, so we need to process successors before predecessors.
    # Standard topo sort gives predecessors before successors.
    # So we can compute PW in reverse of the topo sort order.
    
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                
    # If topo_order length < n, there is a cycle. But ALB instances are DAGs.
    # Initialize PW with 0
    pw = np.zeros(n, dtype=float)
    
    # Process in reverse topological order
    for u in reversed(topo_order):
        # PW_u = time_u + sum(PW_v for v in successors[u])
        total_succ_pw = 0.0
        for v in successors[u]:
            total_succ_pw += pw[v]
        pw[u] = times[u] + total_succ_pw
        
    return pw
