def construct():
    # Grötzsch graph edges
    # Vertices 0-4 form a cycle C5
    # Vertices 5-9 are copies of 0-4
    # Vertex 10 is the top vertex
    # Standard Mycielski construction on C5:
    # 1. Cycle edges: (0,1), (1,2), (2,3), (3,4), (4,0)
    # 2. (v, w') edges for (v,w) in E:
    #    (0,1) -> (0,6), (1,5)
    #    (1,2) -> (1,7), (2,6)
    #    (2,3) -> (2,8), (3,7)
    #    (3,4) -> (3,9), (4,8)
    #    (4,0) -> (4,5), (0,9)
    # 3. (v', u) edges for v in V: (5,10), (6,10), (7,10), (8,10), (9,10)
    
    edges_grotzsch = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        [0, 6], [1, 5],
        [1, 7], [2, 6],
        [2, 8], [3, 7],
        [3, 9], [4, 8],
        [4, 5], [0, 9],
        [5, 10], [6, 10], [7, 10], [8, 10], [9, 10]
    ]
    
    n_base = 11
    adj = [[False]*n_base for _ in range(n_base)]
    for u, v in edges_grotzsch:
        adj[u][v] = True
        adj[v][u] = True
        
    # Find Maximum Independent Set (MIS) in Grötzsch graph
    # Using bitmask enumeration for n=11 (2^11 = 2048, very fast)
    max_size = 0
    max_mask = 0
    
    for mask in range(1 << n_base):
        is_indep = True
        for u in range(n_base):
            if mask & (1 << u):
                # Check if any neighbor v > u is also in mask
                for v in range(u+1, n_base):
                    if (mask & (1 << v)) and adj[u][v]:
                        is_indep = False
                        break
            if not is_indep:
                break
        
        if is_indep:
            size = bin(mask).count('1')
            if size > max_size:
                max_size = size
                max_mask = mask
                
    # The MIS vertices
    mis_vertices = [i for i in range(n_base) if max_mask & (1 << i)]
    
    # Add 5 new vertices: 11, 12, 13, 14, 15
    # Connect each new vertex to all vertices in the MIS
    # This keeps the graph triangle-free because:
    # 1. New vertices are not connected to each other.
    # 2. MIS vertices are not connected to each other.
    # 3. Edges only exist between New and MIS.
    
    new_edges = []
    for new_v in range(11, 16):
        for old_v in mis_vertices:
            new_edges.append([new_v, old_v])
            
    all_edges = edges_grotzsch + new_edges
    
    return all_edges
