import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 40
    V1 = list(range(20))
    V2 = list(range(20, 40))
    
    edges = []
    
    # Add all bipartite edges between V1 and V2 (K_20,20)
    for u in V1:
        for v in V2:
            edges.append([u, v])
            
    # Add one internal edge in V1 to reach 401 edges
    # This creates 20 triangles (one for each vertex in V2)
    edges.append([0, 1])
    
    # Total edges: 400 + 1 = 401
    # Triangles: 20
    # Max Book: 20 (for edge 0-1)
    
    return edges
