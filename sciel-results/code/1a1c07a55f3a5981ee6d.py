import time
import sys
import random
sys.setrecursionlimit(10000)

def construct():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    best_count = 0
    best_edges = []
    
    def greedy_init():
        nonlocal best_count, best_edges
        rng = random.Random(42)
        for _ in range(50):
            order = all_edges[:]
            rng.shuffle(order)
            adj = [0] * n
            c = [[0]*n for _ in range(n)]
            edges = []
            for u, v in order:
                valid = True
                mask_v = adj[v]
                while mask_v:
                    lsb = mask_v & -mask_v
                    w = lsb.bit_length() - 1
                    mask_v ^= lsb
                    if c[u][w] >= 2:
                        valid = False
                        break
                if not valid: continue
                
                mask_u = adj[u]
                while mask_u:
                    lsb = mask_u & -mask_u
                    w = lsb.bit_length() - 1
                    mask_u ^= lsb
                    if c[v][w] >= 2:
                        valid = False
                        break
                if not valid: continue
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
                
                for w in range(n):
                    if w == u or w == v: continue
                    if (adj[v] >> w) & 1:
                        c[u][w] += 1
                        c[w][u] += 1
                    if (adj[u] >> w) & 1:
                        c[v][w] += 1
                        c[w][v] += 1
                        
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]

    greedy_init()
    
    # Sort edges for B&B
    deg = [0]*n
    for u, v in best_edges:
        deg[u] += 1
        deg[v] += 1
    all_edges.sort(key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
    
    start_time = time.time()
    time_limit = 115.0
    
    def solve_with_path(idx, adj, c, count, path):
        nonlocal best_count, best_edges
        
        if time.time() - start_time > time_limit:
            return
            
        if count > best_count:
            best_count = count
            best_edges = path[:]
            
        if idx == num_edges:
            return
            
        if count + (num_edges - idx) <= best_count:
            return
            
        u, v = all_edges[idx]
        
        # Try Include
        valid = True
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            mask_v ^= lsb
            if c[u][w] >= 2:
                valid = False
                break
        if valid:
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                w = lsb.bit_length() - 1
                mask_u ^= lsb
                if c[v][w] >= 2:
                    valid = False
                    break
                    
        if valid:
            c_new = [row[:] for row in c]
            adj_new = adj[:]
            adj_new[u] |= (1 << v)
            adj_new[v] |= (1 << u)
            
            for w in range(n):
                if w == u or w == v: continue
                if (adj_new[v] >> w) & 1:
                    c_new[u][w] += 1
                    c_new[w][u] += 1
                if (adj_new[u] >> w) & 1:
                    c_new[v][w] += 1
                    c_new[w][v] += 1
            
            path.append((u, v))
            solve_with_path(idx + 1, adj_new, c_new, count + 1, path)
            path.pop()
            
        # Try Exclude
        solve_with_path(idx + 1, adj, c, count, path)

    init_adj = [0] * n
    init_c = [[0]*n for _ in range(n)]
    solve_with_path(0, init_adj, init_c, 0, [])
    
    def is_valid(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common.bit_count() >= 3:
                    return False
        return True
        
    if not is_valid(best_edges):
        return [[i, (i+1)%20] for i in range(20)]
        
    return [[u, v] for u, v in best_edges]
