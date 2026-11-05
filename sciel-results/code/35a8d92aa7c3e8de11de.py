def construct():
    n = 35
    # Vertices 0..31 form 8 disjoint K4s
    # Vertices 32, 33, 34 are the extra vertices
    
    # H is the complement of G
    # H consists of:
    # 1. 8 disjoint K4s
    # 2. Extra vertices 32, 33, 34 connected to one "hub" in each K4
    
    # Let's define the hubs as the first vertex of each K4
    # K4 0: 0,1,2,3 -> Hub 0
    # K4 1: 4,5,6,7 -> Hub 4
    # ...
    # K4 7: 28,29,30,31 -> Hub 28
    
    hubs = [0, 4, 8, 12, 16, 20, 24, 28]
    extras = [32, 33, 34]
    
    h_edges = set()
    
    # 1. Add edges for the 8 K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                h_edges.add((min(u, v), max(u, v)))
                
    # 2. Add edges from extras to hubs
    for x in extras:
        for h in hubs:
            h_edges.add((min(x, h), max(x, h)))
            
    # Now construct G (complement of H)
    g_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in h_edges:
                g_edges.append([u, v])
                
    return {'n': n, 'edges': g_edges}
