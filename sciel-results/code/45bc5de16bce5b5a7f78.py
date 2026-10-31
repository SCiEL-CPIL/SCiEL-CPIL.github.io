def luw_construct(k, g, deadline):
    """LUW construction: k-regular bipartite graph with girth >= g."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    # Rule: for c=2..d, rule[c] = (a, b)
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d+1):
        rule[c] = (1, c-2)  # Simple rule
    
    # Generate all vertices
    # Left: (0, x) where x in F_q^d
    # Right: (1, x) where x in F_q^d
    
    def gen_tuples():
        for i in range(q ** d):
            t = []
            x = i
            for j in range(d):
                t.append(x % q)
                x //= q
            yield tuple(t)
    
    # Build adjacency
    # For left vertex p, neighbors are right vertices l such that:
    # l[c] - p[c] = l[a] * p[b] for c=2..d
    # Given p and l[1], we can compute l[c] for c=2..d
    
    edges = []
    vertex_id = {}
    
    # Assign IDs
    idx = 0
    for x in gen_tuples():
        vertex_id[(0, x)] = idx
        idx += 1
    for x in gen_tuples():
        vertex_id[(1, x)] = idx
        idx += 1
    
    n = idx
    
    # Build edges from left side
    for p in gen_tuples():
        p_id = vertex_id[(0, p)]
        for l1 in range(q):
            l = [0] * d
            l[0] = l1  # Wait, indexing...
            
            # Let me use 1-based indexing for the tuple
            # p = (p_1, ..., p_d), l = (l_1, ..., l_d)
            # For c=2..d: l_c = p_c + l_a * p_b where (a,b) = rule[c]
            
            l = [0] * (d+1)  # 1-indexed
            l[1] = l1
            for c in range(2, d+1):
                a, b = rule[c]
                l[c] = (p[c-1] + l[a] * p[b-1]) % q
            
            l_tuple = tuple(l[1:])
            l_id = vertex_id[(1, l_tuple)]
            edges.append((p_id, l_id))
    
    return n, edges
