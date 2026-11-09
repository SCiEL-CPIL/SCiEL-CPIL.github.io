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
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    min_cost = float('inf')
    best_edge = None
    
    # Iterate over all non-edges
    for u in range(31):
        for v in range(u + 1, 31):
            if (u, v) in edges:
                continue
            # Count edges between N(u) and N(v)
            # Note: u and v are not in N(u) or N(v) respectively in a simple graph without loops
            # But we must be careful if u is in N(v) or v is in N(u) - they are not, since (u,v) is a non-edge.
            count = 0
            for w in neighbors[u]:
                for x in neighbors[v]:
                    if (w, x) in edges or (x, w) in edges:
                        count += 1
            # Wait, iterating all pairs in N(u) x N(v) is O(d^2). d~6. 36 ops.
            # 31*30/2 = 465 non-edges. 465 * 36 ~ 16k ops. Very fast.
            
            if count < min_cost:
                min_cost = count
                best_edge = (u, v)
                
    if best_edge:
        edges.add(best_edge)
        
    # Convert to list of lists
    return [list(e) for e in edges]
