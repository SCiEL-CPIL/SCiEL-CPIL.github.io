import time

def count_distinct_cycle_lengths(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    lengths = set()
    
    # Use DFS to find cycles
    # To avoid counting the same cycle multiple times, 
    # we only start DFS from the smallest node in the cycle.
    # This requires that we only visit nodes >= start in the path?
    # No, if we start at 'start', we can only go to nodes > start, 
    # and close the cycle to 'start'.
    
    for start in range(n):
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        def dfs(u):
            if len(lengths) == 9: # Max distinct lengths 4..12 is 9
                return
            for v in adj[u]:
                if v < start:
                    continue
                if v == start:
                    if len(path) >= 4:
                        lengths.add(len(path))
                elif not visited[v]:
                    if len(path) < 12:
                        visited[v] = True
                        path.append(v)
                        dfs(v)
                        path.pop()
                        visited[v] = False
        
        dfs(start)
        
    return len(lengths)

def construct():
    # Two disjoint K3,3 components
    # Vertices 0-5 form K3,3
    # Vertices 6-11 form K3,3
    
    edges = []
    
    # Component 1: 0,1,2 vs 3,4,5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Component 2: 6,7,8 vs 9,10,11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    # Verify constraints
    # Degree: Each vertex in K3,3 has degree 3. Correct.
    # Triangle-free: K3,3 is bipartite. Correct.
    # Cycle: K3,3 has 4-cycles. Correct.
    # Distinct lengths: {4, 6}. Count = 2.
    
    return edges
