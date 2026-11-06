def construct():
    import time
    start = time.time()
    DEADLINE = 110
    
    q = 7
    # Enumerate all 57 points of PG(2,7)
    points = []
    for x2 in range(q):
        for x3 in range(q):
            points.append((1, x2, x3))
    for x3 in range(q):
        points.append((0, 1, x3))
    points.append((0, 0, 1))
    
    n_total = len(points)  # 57
    assert n_total == 57, f"Expected 57 points, got {n_total}"
    
    # Build adjacency
    def dot(p, q_):
        return (p[0]*q_[0] + p[1]*q_[1] + p[2]*q_[2]) % q
    
    adj = [[False]*n_total for _ in range(n_total)]
    for i in range(n_total):
        for j in range(i+1, n_total):
            if dot(points[i], points[j]) == 0:
                adj[i][j] = adj[j][i] = True
    
    # Select 39 vertices
    n = 39
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
