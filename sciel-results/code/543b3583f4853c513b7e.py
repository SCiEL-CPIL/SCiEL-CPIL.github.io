import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s margin to be safe
    
    n = 20
    
    adj = np.zeros((n, n), dtype=int)
    common = np.zeros((n, n), dtype=int)
    edge_count = 0
    
    def can_add(u, v):
        # Check if adding (u, v) violates K2,3
        # Neighbors of v
        for x in range(n):
            if adj[v][x]:
                if common[u][x] >= 2:
                    return False
        # Neighbors of u
        for y in range(n):
            if adj[u][y]:
                if common[v][y] >= 2:
                    return False
        return True

    def add_edge(u, v):
        nonlocal edge_count
        adj[u][v] = 1
        adj[v][u] = 1
        edge_count += 1
        for x in range(n):
            if adj[v][x]:
                common[u][x] += 1
                common[x][u] += 1
        for y in range(n):
            if adj[u][y]:
                common[v][y] += 1
                common[y][v] += 1

    def remove_edge(u, v):
        nonlocal edge_count
        adj[u][v] = 0
        adj[v][u] = 0
        edge_count -= 1
        for x in range(n):
            if adj[v][x]:
                common[u][x] -= 1
                common[x][u] -= 1
        for y in range(n):
            if adj[u][y]:
                common[v][y] -= 1
                common[y][v] -= 1

    # Greedy Initialization
    vertices = list(range(n))
    random.shuffle(vertices)
    edges_to_try = [(u, v) for i, u in enumerate(vertices) for v in vertices[i+1:]]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if can_add(u, v):
            add_edge(u, v)
            
    best_edges = edge_count
    best_adj = adj.copy()
    best_common = common.copy()
    
    # Local Search
    while time.time() < deadline:
        # Try to add any missing edge
        improved = False
        candidates = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if can_add(u, v):
                add_edge(u, v)
                improved = True
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_adj = adj.copy()
                    best_common = common.copy()
                    
        if not improved:
            # Stuck, try a kick
            existing = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u][v]]
            if existing:
                ru, rv = random.choice(existing)
                remove_edge(ru, rv)
            else:
                break

    # Final greedy pass on best solution
    adj_final = best_adj.copy()
    common_final = best_common.copy()
    
    # We need to re-enable can_add/add_edge to use adj_final/common_final
    # Or just do a simple check
    def can_add_final(u, v, adj_f, common_f):
        for x in range(n):
            if adj_f[v][x]:
                if common_f[u][x] >= 2:
                    return False
        for y in range(n):
            if adj_f[u][y]:
                if common_f[v][y] >= 2:
                    return False
        return True

    def add_edge_final(u, v, adj_f, common_f):
        adj_f[u][v] = 1
        adj_f[v][u] = 1
        for x in range(n):
            if adj_f[v][x]:
                common_f[u][x] += 1
                common_f[x][u] += 1
        for y in range(n):
            if adj_f[u][y]:
                common_f[v][y] += 1
                common_f[y][v] += 1

    missing_final = [(u, v) for u in range(n) for v in range(u+1, n) if not adj_final[u][v]]
    random.shuffle(missing_final)
    for u, v in missing_final:
        if can_add_final(u, v, adj_final, common_final):
            add_edge_final(u, v, adj_final, common_final)

    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj_final[u][v]:
                edges.append([u, v])
                
    return edges
