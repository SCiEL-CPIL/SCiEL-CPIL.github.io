"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj, u, v):
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True
    
    # Mycielski(C7): 15 vertices, 28 edges, triangle-free, 4-chromatic
    edges = []
    for i in range(7):
        edges.append([i, (i + 1) % 7])
    for i in range(7):
        u = i
        w = (i + 1) % 7
        edges.append([u, 7 + w])
        edges.append([w, 7 + u])
    for i in range(7):
        edges.append([7 + i, 14])
    
    adj = build_adj(edges)
    
    existing = set()
    for u, v in edges:
        existing.add((min(u,v), max(u,v)))
    
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in existing:
                candidates.append((i, j))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx, (u, v) in enumerate(candidates):
            if (u, v) in existing:
                continue
            if is_triangle_free(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                existing.add((u, v))
                candidates.remove((u, v))
                improved = True
                break
    
    return edges

# EVOLVE-BLOCK-END
