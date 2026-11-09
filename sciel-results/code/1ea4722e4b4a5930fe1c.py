import numpy as np
import time

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor lists
    # predecessors is a list of lists? Or a list of arrays?
    # The prompt says "instance has times, predecessors, cycle_time".
    # Usually predecessors[i] is a list of indices of predecessors of i.
    
    successors = [[] for _ in range(n)]
    for i in range(n):
        if i < len(predecessors):
            for p in predecessors[i]:
                successors[p].append(i)
                
    # Calculate Bottom Level
    # We need reverse topological order.
    # Since we don't have the full graph structure easily for topological sort without indegrees,
    # and n is small (30 tasks), we can just iterate until convergence or use a simple stack.
    # Actually, since it's a DAG, we can compute BL by repeatedly updating.
    # Or better: Topological sort.
    
    # Let's do a simple topological sort to get order.
    # Indegree
    indegree = [0] * n
    for i in range(n):
        if i < len(predecessors):
            for p in predecessors[i]:
                indegree[i] += 1
                
    # Wait, predecessors[i] is the list of predecessors OF i.
    # So edges are p -> i.
    # Indegree of i is len(predecessors[i]).
    
    # Let's re-verify the structure.
    # Standard format: predecessors is a list where predecessors[i] contains indices j such that j must be done before i.
    
    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue = deque()
    for i in range(n):
        if len(predecessors[i]) == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            # We need to decrement indegree of v
            # But we didn't store indegree explicitly in a mutable way for Kahn's
            # Let's just use a simple DFS or BFS for topological sort.
            pass
            
    # Simpler for n=30:
    # Compute BL by dynamic programming in reverse topological order.
    # We can get topological order by sorting nodes by their "level" or just using a stack.
    # Since n is small, we can just do a simple iterative relaxation or DFS.
    
    # Let's use DFS to get reverse topological order.
    visited = [False] * n
    temp_stack = []
    
    def dfs(u):
        visited[u] = True
        for v in successors[u]:
            if not visited[v]:
                dfs(v)
        temp_stack.append(u)
        
    for i in range(n):
        if not visited[i]:
            dfs(i)
            
    # temp_stack is now in reverse topological order (sinks first)
    # So we can compute BL directly.
    bl = np.zeros(n)
    for u in temp_stack:
        max_succ_bl = 0
        for v in successors[u]:
            if bl[v] > max_succ_bl:
                max_succ_bl = bl[v]
        bl[u] = times[u] + max_succ_bl
        
    # Priority is BL
    # To break ties, add a small fraction of time?
    # BL is integer-like if times are integers.
    # Let's just return BL.
    return bl
