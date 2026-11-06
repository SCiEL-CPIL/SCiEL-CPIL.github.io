import time
import random
import math

def construct():
    n = 20
    deadline = time.time() + 100  # Leave margin for 120s limit
    
    # Helper to get bit count
    def popcount(x):
        return bin(x).count("1")
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # For each edge, precompute the list of (a, b) pairs that would violate K2,3 if both (a,b) and (u,v) are present
    # A K2,3 violation occurs when two vertices have 3 common neighbors.
    # If we add edge (u,v), it creates a K2,3 if there exist a,b such that:
    # - a and b are both common neighbors of u and v (so |N(u) ∩ N(v)| >= 3)
    # - OR a is in N(u) and b is in N(v) and a,b have 2 common neighbors already
    # Actually, the condition is: no two vertices have 3 common neighbors.
    # When adding (u,v), we need to check:
    # 1. |N(u) ∩ N(v)| <= 2 (u and v can't have 3 common neighbors)
    # 2. For any w in N(u) (w != v), |N(w) ∩ N(v)| <= 1 (w and v can't have 2 common neighbors plus u)
    #    Wait, N(w) ∩ N(v) includes u if u is in N(w) and u is in N(v). But u is in N(v) only if (u,v) is added.
    #    So after adding (u,v), u becomes a neighbor of v. So for w in N(u), w and v have common neighbors N(w) ∩ N(v) ∪ {u} if u in N(w).
    #    Actually, u is in N(w) since w in N(u). And u will be in N(v) after adding (u,v).
    #    So common neighbors of w and v after adding (u,v) = (N(w) ∩ N(v)) ∪ {u} if u in N(w) and u in N(v).
    #    Since w in N(u), u in N(w). And after adding (u,v), u in N(v).
    #    So common neighbors of w and v = (N(w) ∩ N(v)) ∪ {u}.
    #    We need this to have size <= 2, so |N(w) ∩ N(v)| <= 1 if u is not in N(v) before adding, which it isn't.
    #    Wait, N(w) ∩ N(v) doesn't include u unless u in N(v). Before adding (u,v), u is not in N(v).
    #    So after adding, common neighbors of w and v = (N(w) ∩ N(v)) ∪ {u} = N(w) ∩ N(v) ∪ {u}.
    #    Size = |N(w) ∩ N(v)| + 1 (since u not in N(w)∩N(v) before).
    #    We need this <= 2, so |N(w) ∩ N(v)| <= 1.
    #    Similarly for w in N(v), |N(w) ∩ N(u)| <= 1.
    
    # This is complex to precompute. Let's use a different approach: local search.
    
    best_edges = []
    best_count = 0
    
    # Start with a random greedy construction
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
                    w = i
                    if popcount(adj[w] & adj[v]) > 1:
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
                    w = i
                    if popcount(adj[w] & adj[u]) > 1:
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
    
    # Local search: try to add edges by removing some existing edges first
    def can_add_edge(adj, u, v):
        # Check if edge (u,v) can be added to adj without creating K2,3
        if adj[u] & (1 << v):
            return True  # Already present
        
        # Check 1: |N(u) ∩ N(v)| <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        # Check 2: For all w in N(u), |N(w) ∩ N(v) ∪ {u}| <= 2
        # i.e., |N(w) ∩ N(v)| <= 1 (since u not in N(v) before)
        mask_u = adj[u]
        i = 0
        while mask_u:
            if mask_u & 1:
                w = i
                if popcount(adj[w] & adj[v]) > 1:
                    return False
            mask_u >>= 1
            i += 1
        
        # Check 3: For all w in N(v), |N(w) ∩ N(u) ∪ {v}| <= 2
        # i.e., |N(w) ∩ N(u)| <= 1
        mask_v = adj[v]
        i = 0
        while mask_v:
            if mask_v & 1:
                w = i
                if popcount(adj[w] & adj[u]) > 1:
                    return False
            mask_v >>= 1
            i += 1
        
        return True
    
    def local_search(adj, edges):
        # Try to improve by adding missing edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            
            for u, v in edge_list:
                if adj[u] & (1 << v):
                    continue  # Already present
                
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
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
            # Reconstruct adj from best_edges
            adj_best = [0] * n
            for u, v in best_edges:
                adj_best[u] |= (1 << v)
                adj_best[v] |= (1 << u)
            
            adj_best, edges_best = local_search(adj_best, best_edges[:])
            
            if len(edges_best) > best_count:
                best_count = len(edges_best)
                best_edges = edges_best[:]
            
            # Update adj_best for next iteration
            adj_best = [0] * n
            for u, v in best_edges:
                adj_best[u] |= (1 << v)
                adj_best[v] |= (1 << u)
    
    # Convert list of tuples to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
