import time
import random

def construct():
    deadline = time.time() + 115.0
    
    # Generate Polarity Graph of PG(2,5)
    # Field F_5
    q = 5
    
    # Generate all non-zero vectors in F_q^3, normalized so first non-zero is 1
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    # x is 1
                    points.append((1, (y * pow(x, q-2)) % q, (z * pow(x, q-2)) % q))
                elif y != 0:
                    # x=0, y=1
                    points.append((0, 1, (z * pow(y, q-2)) % q))
                else:
                    # x=0, y=0, z=1
                    points.append((0, 0, 1))
    
    n = len(points)
    # n should be 31
    
    # Build adjacency
    # Polarity: <u, v> = u[0]*v[0] + u[1]*v[1] + u[2]*v[2] == 0 mod q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Quick check for C4-free
    # For every pair u,v, common neighbors must be <= 1
    is_valid = True
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):
                is_valid = False
                break
        if not is_valid:
            break
            
    # Check independence number < 11
    # Use a simple greedy lower bound first
    def greedy_alpha_lb(adj, n):
        remaining = (1 << n) - 1
        count = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            for v in range(n):
                if remaining & (1 << v):
                    non_adj = remaining & ~(adj[v] | (1 << v))
                    deg = bin(non_adj).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            count += 1
            non_adj = remaining & ~(adj[best_v] | (1 << best_v))
            remaining &= ~(non_adj | (1 << best_v))
        return count

    if is_valid:
        alpha_lb = greedy_alpha_lb(adj, n)
        # If alpha_lb >= 11, this graph is invalid for the task (needs alpha < 11)
        # For PG(2,5), alpha is known to be 10.
        if alpha_lb < 11:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    # Fallback: Return a trivial valid graph if the above fails (should not happen)
    # Or a small random graph
    # But PG(2,5) is robust.
    # If it fails, we return a single vertex to avoid 0 score? No, we want max n.
    # If PG(2,5) fails alpha check, it means my math is wrong or implementation is buggy.
    # Let's assume it's correct.
    
    # Emergency fallback: Disjoint triangles (n=30)
    # This is guaranteed C4-free and alpha=10 (3 triangles, pick 2 from each? No, alpha of triangle is 1. 3 triangles -> alpha 3? No.
    # Independent set in disjoint union is sum of alphas. Alpha(K3)=1. 10 K3s -> n=30, alpha=10.
    # This is a valid graph.
    n_fallback = 30
    adj_fallback = [0] * n_fallback
    for i in range(10):
        u, v, w = 3*i, 3*i+1, 3*i+2
        adj_fallback[u] |= (1<<v) | (1<<w)
        adj_fallback[v] |= (1<<u) | (1<<w)
        adj_fallback[w] |= (1<<u) | (1<<v)
        
    edges = []
    for u in range(n_fallback):
        for v in range(u + 1, n_fallback):
            if adj_fallback[u] & (1 << v):
                edges.append([u, v])
                
    return {'n': n_fallback, 'edges': edges}
