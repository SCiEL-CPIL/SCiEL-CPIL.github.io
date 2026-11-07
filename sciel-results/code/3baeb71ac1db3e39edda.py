import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 16
    # Complete bipartite graph K_{8,8} achieves 64 edges with diameter 2.
    # In any bipartite graph, adjacent vertices (across parts) have no common neighbor.
    # Thus, removing any edge (u,v) makes dist(u,v) = 3 (via a length-3 path),
    # increasing the diameter from 2 to 3. Every edge is essential.
    
    half = n // 2
    set_a = list(range(half))
    set_b = list(range(half, n))
    
    edges = []
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return edges
