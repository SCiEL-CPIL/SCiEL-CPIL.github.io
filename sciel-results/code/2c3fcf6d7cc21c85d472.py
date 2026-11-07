import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Construct two disjoint K_{3,3} components.
    # K_{3,3} is bipartite, 3-regular, girth 4.
    # Cycle lengths in K_{3,3} are 4 and 6.
    # Total distinct cycle lengths: {4, 6} -> Count = 2.
    
    edges = []
    
    # Component 1: Vertices 0-5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Component 2: Vertices 6-11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])

    # Convert to Python int lists explicitly to satisfy type constraints
    final_edges = [[int(u), int(v)] for u, v in edges]
    
    # Quick verification (optional but safe)
    n = 12
    deg = [0] * n
    adj = [set() for _ in range(n)]
    for e in final_edges:
        u, v = e
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        
    if min(deg) < 3:
        return []
        
    # Check triangle-free
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if adj[u] & adj[v]:
                    return []
                    
    if time.time() > deadline:
        return final_edges

    return final_edges
