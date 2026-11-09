import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    q = 5
    n = q * q + q + 1  # 31
    
    # Build polarity graph of PG(2, q)
    # Points are 1D subspaces of GF(q)^3
    # Generate all points
    points = []
    for x in range(q):
        for y in range(q):
            points.append([x, y, 1])
    for x in range(q):
        points.append([x, 1, 0])
    points.append([1, 0, 0])
    
    # Normalize: make first nonzero coordinate = 1
    def normalize(p):
        for i in range(3):
            if p[i] != 0:
                inv = pow(p[i], q - 2, q)  # modular inverse in GF(q)
                return [int((p[j] * inv) % q) for j in range(3)]
        return p
    
    norm_points = [normalize(p) for p in points]
    
    # Remove duplicates
    seen = set()
    unique_points = []
    for p in norm_points:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    assert len(unique_points) == n, f"Expected {n} points, got {len(unique_points)}"
    
    # Polarity matrix: M = diag(1, 1, -1) mod q
    # Polar of point p is line {x : p^T M x = 0}
    # Point p is on polar of q if p^T M q = 0
    
    def polar_relation(p, r):
        # Check if p is on the polar line of r
        # p^T M r = 0 where M = diag(1, 1, -1)
        val = (p[0] * r[0] + p[1] * r[1] - p[2] * r[2]) % q
        return val == 0
    
    # Build adjacency
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if polar_relation(unique_points[i], unique_points[j]) or polar_relation(unique_points[j], unique_points[i]):
                edges.add((i, j))
    
    # For a polarity, the relation is symmetric, so we just need one check
    # Actually for a polarity, p is on polar of q iff q is on polar of p
    # So we just need polar_relation(unique_points[i], unique_points[j])
    
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if polar_relation(unique_points[i], unique_points[j]):
                edges.add((i, j))
    
    print(f"Polarity graph edges: {len(edges)}", file=__import__('sys').stderr)
    
    if len(edges) != 90:
        # Fallback: use a different construction
        # Try the standard polarity graph construction
        pass
    
    # Now find the best edge to add to minimize 4-cycles
    # For each non-edge (u,v), count length-3 paths between u and v
    # which equals the number of 4-cycles created by adding (u,v)
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # For each non-edge, count 4-cycles that would be created
    # A 4-cycle u-v-x-w-u is created if v-x, x-w, w-u are edges
    # So for non-edge (u,v), count = number of (x,w) with x in adj[v], w in adj[u], x-w in edges, x != w
    
    best_edge = None
    best_count = float('inf')
    
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges and (j, i) not in edges:
                non_edges.append((i, j))
    
    for u, v in non_edges:
        count = 0
        for x in adj[v]:
            for w in adj[u]:
                if x != w and (min(x, w), max(x, w)) in edges:
                    count += 1
        if count < best_count:
            best_count = count
            best_edge = (u, v)
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        # Return what we have - just the polarity graph plus first non-edge
        if best_edge is None:
            best_edge = non_edges[0] if non_edges else (0, 1)
    
    final_edges = list(edges)
    final_edges.append([best_edge[0], best_edge[1]])
    
    return final_edges
