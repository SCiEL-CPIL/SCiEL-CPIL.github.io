import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Field F_5
    q = 5
    
    # Generate all points in PG(2, 5)
    # Points are non-zero vectors in F_5^3, identified up to scalar multiplication.
    # We normalize by making the first non-zero coordinate 1.
    points = []
    for x0 in range(q):
        for x1 in range(q):
            for x2 in range(q):
                if x0 == 0 and x1 == 0 and x2 == 0:
                    continue
                # Normalize
                if x0 != 0:
                    inv_x0 = pow(x0, q-2, q)
                    p = (1, (x1 * inv_x0) % q, (x2 * inv_x0) % q)
                elif x1 != 0:
                    inv_x1 = pow(x1, q-2, q)
                    p = (0, 1, (x2 * inv_x1) % q)
                else:
                    inv_x2 = pow(x2, q-2, q)
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    n = len(points)
    if n != 31:
        # Fallback if construction fails, though it shouldn't
        points = [(i, 0, 0) for i in range(1, 5)] + [(0, i, 0) for i in range(1, 5)] + [(0, 0, i) for i in range(1, 5)]
        # This fallback is weak, but the above should yield 31 points
        # Actually, let's just use a simple list if the generation is tricky
        # But the generation above is standard.
        pass

    # Map points to indices 0..30
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Build adjacency matrix
    # Polarity: u . v = 0
    adj = [[0]*n for _ in range(n)]
    
    for i in range(n):
        u = points[i]
        for j in range(i+1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i][j] = 1
                adj[j][i] = 1
                
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    # Check edge count
    # The polarity graph of PG(2,5) with dot product should have 90 edges.
    # If it has more or less, we need to adjust.
    # However, for q=5, the number of edges is exactly 90.
    
    if len(edges) > 90:
        # Remove random edges to get to 90? This might break C4-free property?
        # No, removing edges preserves C4-free.
        # But we want to start with a C4-free graph.
        # If we have >90, we remove edges.
        while len(edges) > 90:
            edges.pop()
            # Update adj? Not strictly necessary if we just pick best edge from remaining non-edges
            # But we need to know which edges are present.
            # Let's rebuild adj from edges for correctness if we modify it.
            adj = [[0]*n for _ in range(n)]
            for e in edges:
                adj[e[0]][e[1]] = 1
                adj[e[1]][e[0]] = 1
    elif len(edges) < 90:
        # Add edges that don't create C4?
        # This is harder. But theoretically it should be 90.
        pass

    # Rebuild adj to be safe
    adj = [[0]*n for _ in range(n)]
    for e in edges:
        adj[e[0]][e[1]] = 1
        adj[e[1]][e[0]] = 1
        
    # Find the non-edge that creates the fewest new 4-cycles
    # New 4-cycles involving edge (u,v) correspond to paths of length 3 between u and v in G.
    # Path u-a-b-v.
    # Count = sum over a in N(u) of |N(a) \cap N(v)| excluding u and v?
    # Actually, we just need to count pairs (a,b) such that adj[u][a]=1, adj[a][b]=1, adj[b][v]=1.
    
    min_cost = float('inf')
    best_edge = None
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i+1, n):
            if adj[i][j] == 0:
                # Count paths of length 3 from i to j
                count = 0
                for a in range(n):
                    if adj[i][a]:
                        for b in range(n):
                            if adj[a][b] and adj[b][j]:
                                count += 1
                if count < min_cost:
                    min_cost = count
                    best_edge = [i, j]
                    
    if best_edge is None:
        # Should not happen
        best_edge = [0, 1]
        
    edges.append(best_edge)
    
    return edges

# EVOLVE-BLOCK-END
