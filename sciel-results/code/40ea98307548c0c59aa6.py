import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Pre-generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def has_c6(u, v, adj):
        nu = adj[u]
        nv = adj[v]
        
        a_mask = nu
        while a_mask:
            a = (a_mask & -a_mask).bit_length() - 1
            a_mask &= a_mask - 1
            
            na = adj[a]
            b_mask = na & ~(1 << u)
            
            while b_mask:
                b = (b_mask & -b_mask).bit_length() - 1
                b_mask &= b_mask - 1
                
                if b == v:
                    continue
                    
                nb = adj[b]
                c_mask = nb & ~(1 << a)
                c_mask &= nv
                
                while c_mask:
                    c = (c_mask & -c_mask).bit_length() - 1
                    c_mask &= c_mask - 1
                    
                    if a != c:
                        return True
        return False

    for _ in range(1000):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        edges = []
        
        # Create a shuffled copy
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            
            if not has_c6(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
            
    return best_edges
