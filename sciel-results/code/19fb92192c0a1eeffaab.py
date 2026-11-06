import random
import time

def construct():
    n = 20
    
    def is_valid(u, v, adj):
        # adj is list of ints
        for w in range(n):
            if w == u or w == v:
                continue
            
            # If v is neighbor of w, then N(u) and N(w) gain a common neighbor v?
            # No, N(u) gains v. So if v in N(w), then v is in N(u) and N(w).
            # So |N(u) cap N(w)| increases by 1.
            # We need new size <= 2. So old size must be <= 1.
            if (adj[v] >> w) & 1:
                common = adj[u] & adj[w]
                if common.bit_count() > 1:
                    return False
            
            # If u is neighbor of w, then N(v) and N(w) gain a common neighbor u.
            if (adj[u] >> w) & 1:
                common = adj[v] & adj[w]
                if common.bit_count() > 1:
                    return False
        
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    best_edges = 0
    best_adj = [0] * n
    
    start_time = time.time()
    deadline = 110.0
    
    # To ensure we find good solutions, we can use multiple strategies
    # 1. Random order greedy
    # 2. Maybe degree-based order?
    
    while time.time() < deadline:
        adj = [0] * n
        
        # Random shuffle
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if is_valid(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        m = count_edges(adj)
        if m > best_edges:
            best_edges = m
            best_adj = adj[:]
            
        # Simple local search: try to add edges in reverse order?
        # Or just restart.
        
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                final_edges.append([i, j])
                
    return final_edges
