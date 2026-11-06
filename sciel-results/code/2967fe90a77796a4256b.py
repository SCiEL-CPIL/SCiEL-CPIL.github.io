import random
import numpy as np

def _is_k4_free(n, adj):
    """Check if graph with adjacency matrix adj on n vertices is K4-free using numpy."""
    # adj is a numpy array of 0s and 1s
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                continue
            common = adj[i] & adj[j]
            cs = np.nonzero(common)[0]
            if len(cs) < 2:
                continue
            for k in cs:
                if np.any(adj[k, cs]):
                    return False
    return True

def _max_triangle_free_induced(n, adj):
    """Exact branch and bound for largest triangle-free induced subgraph using bitmasks."""
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in range(n):
            if adj[i, j]:
                m |= (1 << j)
        masks[i] = m
    
    best = [0]
    
    def is_independent(mask):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            rest = mask ^ lsb
            if masks[idx] & rest:
                return False
            m ^= lsb
        return True

    def dfs(start, current_mask, current_size):
        if current_size + (n - start) <= best[0]:
            return
        if current_size > best[0]:
            best[0] = current_size
            
        for v in range(start, n):
            neigh_in_cur = current_mask & masks[v]
            if is_independent(neigh_in_cur):
                dfs(v + 1, current_mask | (1 << v), current_size + 1)
    
    dfs(0, 0, 0)
    return best[0]

def construct():
    n = 18
    random.seed(42)
    
    # Start with Turan T3(18)
    adj = np.zeros((n, n), dtype=int)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a, b] = 1
                    adj[b, a] = 1
    
    best_score = _max_triangle_free_induced(n, adj)
    best_adj = adj.copy()
    
    # Local Search
    for iteration in range(5000):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        new_adj = adj.copy()
        new_adj[u, v] = 1 - new_adj[u, v]
        new_adj[v, u] = new_adj[u, v]
        
        if not _is_k4_free(n, new_adj):
            continue
            
        score = _max_triangle_free_induced(n, new_adj)
        if score < best_score:
            best_score = score
            best_adj = new_adj
            adj = new_adj
            if best_score <= 7:
                break
        elif score == best_score:
            if random.random() < 0.5:
                adj = new_adj
                best_adj = new_adj
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                edges.append([i, j])
                
    return edges
