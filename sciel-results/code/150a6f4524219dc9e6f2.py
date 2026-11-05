import numpy as np
import time

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    
    # Build successor lists
    # predecessors[i] is a list of indices of predecessors of i
    # We need successors for efficient longest path calculation from end to start
    successors = [[] for _ in range(n)]
    indegree = [0] * n
    
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            indegree[i] += 1
            
    # Topological sort (Kahn's algorithm) to get order
    # We want to process nodes in topological order to compute longest paths from start?
    # No, we want longest path TO THE END.
    # So we should process in REVERSE topological order.
    
    # Let's just use DFS with memoization for simplicity and speed on small N (30).
    # N=30 is very small. O(N^2) or even O(N^3) is fine.
    
    # cp[i] = longest path from i to any sink, including t[i]
    cp = [0.0] * n
    visited = [False] * n
    
    def dfs(u):
        if visited[u]:
            return cp[u]
        visited[u] = True
        max_succ = 0.0
        for v in successors[u]:
            max_succ = max(max_succ, dfs(v))
        cp[u] = times[u] + max_succ
        return cp[u]
        
    for i in range(n):
        if not visited[i]:
            dfs(i)
            
    return np.array(cp)
