def construct():
    # Construct C4-free graph via orthogonal polarity graph on PG(2,7)
    # PG(2,7) has 57 points; take an induced subgraph on 43 vertices
    
    # Enumerate points of PG(2,7) in canonical form
    points = []
    for x in range(7):
        for y in range(7):
            for z in range(7):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 6, 7)
                    points.append((1, (y * inv_x) % 7, (z * inv_x) % 7))
                elif y != 0:
                    inv_y = pow(y, 6, 7)
                    points.append((0, 1, (z * inv_y) % 7))
                else:
                    points.append((0, 0, 1))
    
    points = list(set(points))
    points.sort()
    assert len(points) == 57, f"Expected 57 points, got {len(points)}"
    
    # Orthogonal polarity: p·q = x1*x2 + x2*x3 + x3*x1 (mod 7)
    def dot(p, q):
        return (p[0] * q[1] + p[1] * q[2] + p[2] * q[0]) % 7
    
    # Build adjacency matrix
    adj = [[False] * 57 for _ in range(57)]
    for i in range(57):
        for j in range(i + 1, 57):
            if dot(points[i], points[j]) == 0:
                adj[i][j] = adj[j][i] = True
    
    # Verify C4-free: no two vertices share more than one common neighbor
    for i in range(57):
        for j in range(i + 1, 57):
            common = 0
            for k in range(57):
                if adj[i][k] and adj[j][k]:
                    common += 1
            if common > 1:
                raise ValueError(f"C4 detected: vertices {i} and {j} have {common} common neighbors")
    
    # Take induced subgraph on first 43 vertices
    n = 43
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
