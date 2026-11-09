import numpy as np
from collections import defaultdict, deque
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 10.0  # 10 second deadline, well within 120s
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    # Build successors graph
    succs = defaultdict(list)
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    
    # Compute in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(preds[i])
    
    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        if time.time() > deadline:
            break
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't have all nodes, there's a cycle. Handle by adding remaining nodes.
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Compute critical path (forward) in reverse topological order
    cp = np.zeros(n)
    for u in reversed(topo_order):
        if time.time() > deadline:
            break
        max_succ_cp = 0.0
        for v in succs[u]:
            if cp[v] > max_succ_cp:
                max_succ_cp = cp[v]
        cp[u] = times[u] + max_succ_cp
    
    # Compute number of transitive successors for each node using BFS
    trans_succ_count = np.zeros(n, dtype=int)
    for u in range(n):
        if time.time() > deadline:
            break
        visited = set()
        dq = deque(succs[u])
        while dq:
            v = dq.popleft()
            if v not in visited:
                visited.add(v)
                for w in succs[v]:
                    if w not in visited:
                        dq.append(w)
        trans_succ_count[u] = len(visited)
    
    # Priority: cp * (1 + trans_succ_count) + small LPT tie-breaker
    priority_scores = cp * (1.0 + trans_succ_count) + times * 1e-6
    
    return priority_scores
