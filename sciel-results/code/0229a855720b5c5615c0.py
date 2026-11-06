def construct():
    # P3: vertices 0, 1, 2. Edges (0,1), (1,2)
    edges_p3 = [[0, 1], [1, 2]]
    n_p3 = 3
    
    # Mycielski construction
    def mycielski(edges, n):
        new_edges = []
        # 1. Original edges
        for u, v in edges:
            new_edges.append([u, v])
        
        # 2. Copy vertices: n to 2n-1
        # For each edge (u, v), add (u', v) and (v', u)
        for u, v in edges:
            new_edges.append([n + u, v])
            new_edges.append([n + v, u])
        
        # 3. Top vertex: 2n
        # Connect to all original vertices
        for u in range(n):
            new_edges.append([2 * n, u])
            
        return new_edges, 2 * n + 1

    # G7 = M(P3)
    edges_g7, n_g7 = mycielski(edges_p3, n_p3)
    
    # G15 = M(G7)
    edges_g15, n_g15 = mycielski(edges_g7, n_g7)
    
    # n_g15 is 15. Vertices 0..14.
    # The copy vertices in G15 are 7..13 (since G7 had 7 vertices, copies are 7..13).
    # These form an independent set.
    copy_vertices = list(range(n_g7, 2 * n_g7)) # 7 to 13
    
    # Add vertex 15
    # Connect 15 to all copy_vertices
    for v in copy_vertices:
        edges_g15.append([15, v])
        
    return edges_g15
