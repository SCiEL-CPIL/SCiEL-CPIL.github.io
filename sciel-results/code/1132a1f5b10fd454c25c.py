import random
import time

def construct():
    n = 20
    start_time = time.time()
    timeout = 140  # leave margin for 150s limit
    
    # Precompute all potential edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    m = len(all_edges)  # 190
    
    # Edge index map
    edge_idx = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    
    # Build conflict graph: two edges conflict if they cannot coexist in a K2,3-free graph
    # Two edges (u,v) and (w,x) conflict if:
    # 1. They share a vertex and the other endpoints have 2+ common neighbors among {u,v,w,x}
    # 2. More precisely, we need to check all K2,3 patterns
    
    # Actually, let's think more carefully. A K2,3 has two vertices a,b connected to three common neighbors c,d,e.
    # So edges (a,c), (a,d), (a,e), (b,c), (b,d), (b,e) form a K2,3.
    # The constraint is: no two vertices have 3+ common neighbors.
    # 
    # When we add edge (u,v), it conflicts with existing edges if:
    # - For some w in N(v), |N(u) ∩ N(w)| >= 2 (then adding (u,v) makes v a 3rd common neighbor of u,w)
    # - For some w in N(u), |N(v) ∩ N(w)| >= 2
    # - |N(u) ∩ N(v)| >= 3
    #
    # For the conflict graph approach, we need to determine which pairs of edges cannot both be present.
    # This is complex. Let's use a different approach: use the local search but with better heuristics.
    
    # Use adjacency list representation
    adj = [set() for _ in range(n)]
    
    def can_add(u, v):
        if v in adj[u]:
            return False
            
        common_uv = adj[u] & adj[v]
        if len(common_uv) > 2:
            return False
            
        for w in adj[v]:
            if len(adj[u] & adj[w]) > 1:
                return False
                
        for w in adj[u]:
            if len(adj[v] & adj[w]) > 1:
                return False
                
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)

    def get_edges():
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    def edge_count():
        return sum(len(s) for s in adj) // 2

    # Multi-start local search with multiple random initializations
    best_edges = []
    best_count = 0
    
    # Run multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Reset
        adj = [set() for _ in range(n)]
        
        # Random greedy initialization
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if can_add(u, v):
                add_edge(u, v)
        
        current_count = edge_count()
        
        # Local search: try to add missing edges, then swap
        for iteration in range(1000):
            if time.time() - start_time > timeout:
                break
            
            # Try to add any missing edge
            added = False
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            random.shuffle(missing)
            
            for u, v in missing:
                if can_add(u, v):
                    add_edge(u, v)
                    added = True
            
            if not added:
                # Try swaps: remove edge e1, add edge e2
                all_current = get_edges()
                random.shuffle(all_current)
                
                found_swap = False
                for e1 in all_current:
                    u1, v1 = e1
                    # Try removing e1 and adding a missing edge
                    remove_edge(u1, v1)
                    
                    missing_after = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
                    random.shuffle(missing_after)
                    
                    best_add = None
                    for u2, v2 in missing_after:
                        if can_add(u2, v2):
                            best_add = (u2, v2)
                            break
                    
                    if best_add:
                        add_edge(best_add[0], best_add[1])
                        found_swap = True
                        break
                    else:
                        # Restore
                        add_edge(u1, v1)
                
                if not found_swap:
                    # Stuck, break
                    break
            
            current_count = edge_count()
            
            if current_count > best_count:
                best_count = current_count
                best_edges = get_edges()
    
    # Final verification and return
    if best_edges:
        return best_edges
    else:
        # Fallback: return empty graph
        return []
