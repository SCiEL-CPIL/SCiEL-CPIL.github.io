import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 20
    
    def is_valid_addition(adj, u, v):
        # Check if adding (u,v) keeps graph K3,3-free
        # Constraint: no 3 vertices have 3+ common neighbors
        # Adding (u,v) only affects triples containing u or v
        
        # For triples {u, x, y}: v becomes a common neighbor if v~x and v~y
        # Need: for all x,y != u,v, |N(u)∩N(x)∩N(y)| + [v~x and v~y] <= 2
        # Note: after adding, v ∈ N(u)
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                # Count common neighbors of {u,x,y} in current graph (before adding u-v)
                cnt = 0
                for w in range(n):
                    if w == u or w == x or w == y:
                        continue
                    if adj[w][u] and adj[w][x] and adj[w][y]:
                        cnt += 1
                # After adding (u,v), v is in N(u). v is common neighbor of {u,x,y} iff v~x and v~y
                if adj[v][x] and adj[v][y]:
                    cnt += 1
                if cnt > 2:
                    return False
        
        # For triples {v, x, y}: u becomes a common neighbor if u~x and u~y
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                cnt = 0
                for w in range(n):
                    if w == u or w == x or w == y:
                        continue
                    if adj[w][v] and adj[w][x] and adj[w][y]:
                        cnt += 1
                if adj[u][x] and adj[u][y]:
                    cnt += 1
                if cnt > 2:
                    return False
        
        return True
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    def greedy_build(adj, order):
        for (u, v) in order:
            if not adj[u][v] and is_valid_addition(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
    
    def local_search_add_only(adj):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                if time.time() > deadline:
                    return adj
                for v in range(u+1, n):
                    if not adj[u][v] and is_valid_addition(adj, u, v):
                        adj[u][v] = 1
                        adj[v][u] = 1
                        improved = True
                        break
                if improved:
                    break
        return adj
    
    best_adj = None
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Try different random orders
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        
        adj = [[0]*n for _ in range(n)]
        order = all_edges[:]
        random.shuffle(order)
        
        greedy_build(adj, order)
        adj = local_search_add_only(adj)
        cnt = count_edges(adj)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = [row[:] for row in adj]
        
        if best_count >= 180:
            break
    
    if best_adj is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    return edges
