import random
import time

def construct():
    # Initial graph: 5 disjoint K4s
    edges = set()
    for k in range(5):
        for a in range(4):
            for b in range(a+1, 4):
                edges.add((4*k + a, 4*k + b))
    
    # Adjacency lists for fast lookup
    adj = [set() for _ in range(20)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def is_valid(u, v):
        # Check if adding edge (u,v) creates a K2,3
        # 1. Check pair (u,v): |N(u) N(v)| must be <= 2
        if len(adj[u] & adj[v]) >= 3:
            return False
            
        # 2. Check pairs (u, w) where w in N(v)
        # Adding (u,v) makes v a neighbor of u.
        # So for any w in N(v), if v is in N(w) (which is true since w in N(v) implies v in N(w)),
        # then v becomes a common neighbor of u and w.
        # So we need to check if |N(u) N(w)| + (1 if v in N(u) else 0) <= 2?
        # Wait. N(u) will include v.
        # New common neighbors of (u,w) = (N(u) U {v}) N N(w).
        # Since v in N(w), this is (N(u) N N(w)) U {v} (if v not in N(u) N N(w)).
        # Basically, count = |N(u) N N(w)| + (1 if v not in N(u) N N(w) else 0).
        # But v is in N(w). Is v in N(u)? No, we are adding it.
        # So v is NOT in N(u) currently.
        # So new count = |N(u) N N(w)| + 1.
        # We need |N(u) N N(w)| + 1 <= 2 => |N(u) N N(w)| <= 1.
        
        for w in adj[v]:
            if w == u: continue
            # Check pair (u, w)
            if len(adj[u] & adj[w]) > 1:
                return False
                
        # 3. Check pairs (v, w) where w in N(u)
        for w in adj[u]:
            if w == v: continue
            if len(adj[v] & adj[w]) > 1:
                return False
                
        return True

    # Local Search
    start_time = time.time()
    timeout = 10.0 # Leave margin for 120s limit
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(20) for j in range(i+1, 20)]
    
    # Shuffle for random order
    random.shuffle(all_edges)
    
    added = 0
    for u, v in all_edges:
        if time.time() - start_time > timeout:
            break
        if (u, v) in edges:
            continue
        if is_valid(u, v):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            added += 1
            
    # Second pass? Maybe try again with different shuffle?
    # Or try to remove edges to allow more additions?
    # For now, just greedy addition.
    
    return [list(e) for e in edges]
