import time
import random
from itertools import combinations

def _is_k4_free(n, adj):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                continue
            for k in range(j+1, n):
                if not adj[i][k] or not adj[j][k]:
                    continue
                for l in range(k+1, n):
                    if adj[i][l] and adj[j][l] and adj[k][l]:
                        return False
    return True

def _max_triangle_free_size(n, adj):
    """Find the largest subset of vertices inducing a triangle-free subgraph."""
    # Branch and bound
    best = [0]
    
    def is_triangle_free(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if adj[a][b] and adj[a][c] and adj[b][c]:
                        return False
        return True
    
    # Use a recursive branch and bound
    # State: current set (as a bitmask or list), next vertex to consider
    # For 18 vertices, use bitmask
    
    from functools import lru_cache
    
    # Precompute: for each pair, which vertices form a triangle with them
    # tri_with[i][j] = set of k such that i,j,k form a triangle
    tri_with = [[set() for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                continue
            for k in range(n):
                if k != i and k != j and adj[i][k] and adj[j][k]:
                    tri_with[i][j].add(k)
                    tri_with[j][i].add(k)
    
    def dfs(current_mask, next_v, size):
        if time.time() > deadline[0]:
            return
        if size + (n - next_v) <= best[0]:
            return
        if next_v == n:
            if size > best[0]:
                best[0] = size
            return
        
        # Option 1: don't include next_v
        dfs(current_mask, next_v + 1, size)
        
        # Option 2: include next_v, if it doesn't create a triangle
        v = next_v
        # Check if v can be added: no two vertices in current_mask that form a triangle with v
        # i.e., for all u in current_mask, v not in tri_with[u][v]
        # Actually: for all u in current_mask, not (adj[u][v] and there exists w in current_mask with adj[u][w] and adj[v][w])
        # Simpler: check all pairs in current_mask
        can_add = True
        bits = []
        for u in range(next_v):
            if current_mask & (1 << u):
                bits.append(u)
        for idx in range(len(bits)):
            for jdx in range(idx+1, len(bits)):
                u, w = bits[idx], bits[jdx]
                if v in tri_with[u][w]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            dfs(current_mask | (1 << v), next_v + 1, size + 1)
    
    dfs(0, 0, 0)
    return best[0]

def construct():
    global deadline
    start_time = time.time()
    deadline = [start_time + 140]  # 10s safety margin
    
    n = 18
    random.seed(42)
    
    best_score = n
    best_adj = None
    
    # Initialize with a random K4-free graph
    # Start with empty graph and add edges
    adj = [[False]*n for _ in range(n)]
    
    # Try to build a good graph
    # Strategy: start with a random graph, ensure K4-free, then optimize
    
    def try_local_search(adj_init, iterations=5000):
        nonlocal best_score, best_adj
        adj = [row[:] for row in adj_init]
        
        for it in range(iterations):
            if time.time() > deadline[0] - 30:
                break
            
            # Compute current score
            score = _max_triangle_free_size(n, adj)
            if score < best_score:
                best_score = score
                best_adj = [row[:] for row in adj]
            
            if best_score <= 8:  # Can't do much better
                break
            
            # Try a random move: add or remove an edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            if adj[u][v]:
                # Remove edge
                adj[u][v] = adj[v][u] = False
            else:
                # Try to add edge, check K4-free
                # Adding edge (u,v) creates K4 if there exist x,y such that
                # u-x, u-y, v-x, v-y, x-y all exist
                can_add = True
                for x in range(n):
                    if x == u or x == v or not adj[u][x]:
                        continue
                    for y in range(x+1, n):
                        if y == u or y == v or not adj[v][y] or not adj[x][y]:
                            continue
                        # Check if u,v,x,y form K4
                        if adj[u][v] is False:  # we're about to add it
                            if adj[u][x] and adj[u][y] and adj[v][x] and adj[v][y] and adj[x][y]:
                                can_add = False
                                break
                    if not can_add:
                        break
                
                if can_add:
                    adj[u][v] = adj[v][u] = True
        
        return adj
    
    # Start from several initial configurations
    # 1. Empty graph
    # 2. Random sparse graph
    # 3. Random dense graph (then make K4-free)
    
    # Try from empty
    adj_empty = [[False]*n for _ in range(n)]
    try_local_search(adj_empty, 3000)
    
    # Try from random
    for trial in range(5):
        if time.time() > deadline[0] - 60:
            break
        adj_rand = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.3:
                    adj_rand[i][j] = adj_rand[j][i] = True
        # Make K4-free by removing edges
        changed = True
        while changed and time.time() < deadline[0] - 60:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if not adj_rand[i][j]:
                        continue
                    for k in range(j+1, n):
                        if not adj_rand[i][k] or not adj_rand[j][k]:
                            continue
                        for l in range(k+1, n):
                            if adj_rand[i][l] and adj_rand[j][l] and adj_rand[k][l]:
                                # Remove one edge to break K4
                                adj_rand[i][j] = adj_rand[j][i] = False
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        
        try_local_search(adj_rand, 3000)
    
    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback: complete tripartite
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
