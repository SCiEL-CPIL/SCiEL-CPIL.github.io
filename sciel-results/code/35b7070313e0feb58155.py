def construct():
    # Generate points of PG(2,5)
    points = []
    # Type 1: (1, a, b)
    for a in range(5):
        for b in range(5):
            points.append([1, a, b])
    # Type 2: (0, 1, c)
    for c in range(5):
        points.append([0, 1, c])
    # Type 3: (0, 0, 1)
    points.append([0, 0, 1])
    
    # Map point tuple to index
    point_to_idx = {tuple(p): i for i, p in enumerate(points)}
    
    # Compute adjacency
    # B(x,y) = x1*y1 + x2*y2 + x3*y3 mod 5
    edges = set()
    for i in range(31):
        for j in range(i + 1, 31):
            x = points[i]
            y = points[j]
            val = (x[0]*y[0] + x[1]*y[1] + x[2]*y[2]) % 5
            if val == 0:
                edges.add((i, j))
                
    # We have 90 edges. Need 91.
    # Find non-edge (u, v) that minimizes new 4-cycles.
    # New 4-cycles = number of edges between N(u) and N(v) in the current graph.
    
    # Precompute neighbors
    neighbors = [set() for _ in range(31)]
    # Normalize edges for faster lookup
    norm_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        norm_edges.add((u, v))
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    min_cost = float('inf')
    best_edge = None
    
    # Iterate over all non-edges
    for u in range(31):
        for v in range(u + 1, 31):
            if (u, v) in norm_edges:
                continue
            # Count edges between N(u) and N(v)
            count = 0
            for w in neighbors[u]:
                for x in neighbors[v]:
                    # Check if edge (w,x) exists
                    if w < x:
                        if (w, x) in norm_edges:
                            count += 1
                    else:
                        if (x, w) in norm_edges:
                            count += 1
                            
            if count < min_cost:
                min_cost = count
                best_edge = (u, v)
                
    if best_edge:
        u, v = best_edge
        if u > v:
            u, v = v, u
        norm_edges.add((u, v))
        edges.add((u, v))
        
    # Convert to list of lists
    return [list(e) for e in edges]
