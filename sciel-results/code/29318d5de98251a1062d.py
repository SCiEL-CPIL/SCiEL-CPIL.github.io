import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Leave margin for 120s limit
    
    # Helper to get bit count (fast)
    def popcount(x):
        return x.bit_count()
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # Fast greedy construction
    def greedy_construct():
        adj = [0] * n
        edges = []
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            # Check 1: |N(u) ∩ N(v)| <= 2
            if popcount(adj[u] & adj[v]) > 2:
                continue
            
            # Check 2: For all w in N(u), |N(w) ∩ N(v)| <= 1
            mask_u = adj[u]
            valid = True
            i = 0
            while mask_u:
                if mask_u & 1:
                    if popcount(adj[i] & adj[v]) > 1:
                        valid = False
                        break
                mask_u >>= 1
                i += 1
            if not valid:
                continue
            
            # Check 3: For all w in N(v), |N(w) ∩ N(u)| <= 1
            mask_v = adj[v]
            i = 0
            while mask_v:
                if mask_v & 1:
                    if popcount(adj[i] & adj[u]) > 1:
                        valid = False
                        break
                mask_v >>= 1
                i += 1
            if not valid:
                continue
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
        
        return adj, edges
    
    # Fast can_add_edge check
    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return True
        
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        mask_u = adj[u]
        i = 0
        while mask_u:
            if mask_u & 1:
                if popcount(adj[i] & adj[v]) > 1:
                    return False
            mask_u >>= 1
            i += 1
        
        mask_v = adj[v]
        i = 0
        while mask_v:
            if mask_v & 1:
                if popcount(adj[i] & adj[u]) > 1:
                    return False
            mask_v >>= 1
            i += 1
        
        return True
    
    # Local search: add edges until no more can be added
    def local_search(adj, edges):
        edge_set = set(edges)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            
            for u, v in edge_list:
                if (u, v) in edge_set:
                    continue
                
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
                    edge_set.add((u, v))
                    improved = True
        
        return adj, edges
    
    # Main loop
    while time.time() < deadline:
        adj, edges = greedy_construct()
        adj, edges = local_search(adj, edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        # Try to improve current best with local search
        if time.time() < deadline:
            adj_best = [0] * n
            for u, v in best_edges:
                adj_best[u] |= (1 << v)
                adj_best[v] |= (1 << u)
            
            adj_best, edges_best = local_search(adj_best, best_edges[:])
            
            if len(edges_best) > best_count:
                best_count = len(edges_best)
                best_edges = edges_best[:]
    
    return [[u, v] for u, v in best_edges]
