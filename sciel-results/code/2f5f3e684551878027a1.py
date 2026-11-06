import random
import time

def construct():
    n = 20
    adj = [[False] * n for _ in range(n)]
    cnt = [[[0] * n for _ in range(n)] for __ in range(n)]
    
    def get_idx(i, j, k):
        arr = sorted([i, j, k])
        return arr[0], arr[1], arr[2]

    def is_valid_add(u, v):
        if adj[u][v]:
            return True
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x + 1, n):
                if y == u or y == v: continue
                if adj[v][x] and adj[v][y]:
                    i, j, k = get_idx(u, x, y)
                    if cnt[i][j][k] == 2:
                        return False
                if adj[u][x] and adj[u][y]:
                    i, j, k = get_idx(v, x, y)
                    if cnt[i][j][k] == 2:
                        return False
        return True

    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x + 1, n):
                if y == u or y == v: continue
                if adj[v][x] and adj[v][y]:
                    i, j, k = get_idx(u, x, y)
                    cnt[i][j][k] += 1
                if adj[u][x] and adj[u][y]:
                    i, j, k = get_idx(v, x, y)
                    cnt[i][j][k] += 1

    def remove_edge(u, v):
        adj[u][v] = False
        adj[v][u] = False
        for x in range(n):
            if x == u or x == v: continue
            for y in range(x + 1, n):
                if y == u or y == v: continue
                if adj[v][x] and adj[v][y]:
                    i, j, k = get_idx(u, x, y)
                    cnt[i][j][k] -= 1
                if adj[u][x] and adj[u][y]:
                    i, j, k = get_idx(v, x, y)
                    cnt[i][j][k] -= 1

    def has_violation():
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if cnt[i][j][k] >= 3:
                        return True
        return False

    def fix_violations():
        while has_violation():
            # Find a violating triple
            # Just pick a random edge to remove
            # Better: pick an edge that is part of a violation?
            # For simplicity, random edge removal
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u][v]]
            if not edges: break
            u, v = random.choice(edges)
            remove_edge(u, v)

    # Initialization
    random.seed(42)
    # Start with random graph p=0.5
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                add_edge(u, v)
    
    fix_violations()
    
    start_time = time.time()
    best_edges = sum(1 for u in range(n) for v in range(u + 1, n) if adj[u][v])
    best_graph = [row[:] for row in adj]
    
    stall_counter = 0
    
    while time.time() - start_time < 140:
        # Try to add a random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v: continue
        if u > v: u, v = v, u
        
        if not adj[u][v]:
            if is_valid_add(u, v):
                add_edge(u, v)
                stall_counter = 0
                current_edges = best_edges + 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_graph = [row[:] for row in adj]
            else:
                stall_counter += 1
        
        if stall_counter > 100:
            # Kick: remove a random edge
            edges = [(a, b) for a in range(n) for b in range(a + 1, n) if adj[a][b]]
            if edges:
                a, b = random.choice(edges)
                remove_edge(a, b)
                best_edges -= 1 # This might not be best, but we are exploring
            stall_counter = 0
            
    # Return best graph found
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_graph[u][v]:
                edges_list.append([u, v])
    return edges_list
