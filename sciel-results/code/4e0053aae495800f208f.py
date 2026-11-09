"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    
    # Build the Clebsch graph as starting point
    # Vertices 0..15, edge if Hamming distance is 1 or 4
    adj = [[False]*n for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            xor = u ^ v
            bits = bin(xor).count('1')
            if bits == 1 or bits == 4:
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True
    
    def is_3_colorable(adj):
        # Backtracking 3-coloring
        color = [-1]*n
        def backtrack(idx):
            if idx == n:
                return True
            for c in range(3):
                ok = True
                for v in range(idx):
                    if adj[idx][v] and color[v] == c:
                        ok = False
                        break
                if ok:
                    color[idx] = c
                    if backtrack(idx+1):
                        return True
                    color[idx] = -1
            return False
        return backtrack(0)
    
    # Try to add edges
    # For each non-edge, check if adding it keeps triangle-free and 4-chromatic
    # We'll do multiple passes
    for _ in range(5):  # 5 passes
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    # Check if adding (u,v) creates a triangle
                    creates_triangle = False
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            creates_triangle = True
                            break
                    if not creates_triangle:
                        # Try adding the edge
                        adj[u][v] = adj[v][u] = True
                        # Check if still 4-chromatic (i.e., not 3-colorable)
                        if not is_3_colorable(adj):
                            edges.append([u, v])
                            improved = True
                        else:
                            # Remove the edge
                            adj[u][v] = adj[v][u] = False
    
        if not improved:
            break
    
    return edges
# EVOLVE-BLOCK-END
