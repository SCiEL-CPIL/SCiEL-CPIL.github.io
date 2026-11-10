def construct():
    base_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
                  [5, 1], [5, 4],
                  [6, 0], [6, 2],
                  [7, 1], [7, 3],
                  [8, 2], [8, 4],
                  [9, 3], [9, 0],
                  [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]]
    
    adj = set()
    for u, v in base_edges:
        adj.add((u, v))
        adj.add((v, u))
    
    indep = {5, 6, 7, 8, 9}
    for v in range(11, 16):
        for u in indep:
            adj.add((v, u))
            adj.add((u, v))
    
    def has_common_neighbor(u, v, adj_set):
        for w in range(16):
            if w != u and w != v:
                if (u, w) in adj_set and (v, w) in adj_set:
                    return True
        return False
    
    # For each new vertex, determine which old vertices it can connect to
    # such that no triangle is formed
    # This is equivalent to: the set of new connections for vertex v must be an independent set in G[0..10]
    # AND also must not create triangles with the indep set connections
    
    # Actually, the constraint is: for vertex v in 11-15, the set N(v) ∩ {0..10} must be an independent set
    # because if two vertices u1, u2 in N(v)∩{0..10} are adjacent, then u1-u2-v is a triangle.
    
    # Also, N(v) includes indep = {5,6,7,8,9}, and these must not be adjacent to each other (they aren't).
    # And for any u in N(v)∩{0..10} and any w in indep, if u-w is an edge, that's fine as long as no triangle.
    # Triangle v-u-w: need u-w edge and v-w edge (yes) and v-u edge (yes). So if u is adjacent to any w in indep, triangle!
    
    # So N(v)∩{0..10} must be an independent set AND no vertex in it can be adjacent to any vertex in indep.
    
    # Vertices in 0-4 and 10:
    # 0: adjacent to 6,9 in indep → CANNOT include 0
    # 1: adjacent to 5,7 in indep → CANNOT include 1
    # 2: adjacent to 6,8 in indep → CANNOT include 2
    # 3: adjacent to 7,9 in indep → CANNOT include 3
    # 4: adjacent to 5,8 in indep → CANNOT include 4
    # 10: adjacent to 5,6,7,8,9 in indep → CANNOT include 10
    
    # So none of 0-4 or 10 can be connected to 11-15!
    
    # Therefore, the only edges are the base Grötzsch + 25 indep edges = 45 edges.
    
    all_edges = list(base_edges)
    for v in range(11, 16):
        for u in indep:
            all_edges.append([min(u,v), max(u,v)])
    
    return all_edges
