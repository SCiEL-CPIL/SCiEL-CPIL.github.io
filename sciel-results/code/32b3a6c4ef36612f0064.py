import random
import time

def construct():
    n = 20
    start_time = time.time()
    timeout = 140  # leave margin for 150s limit

    # Precompute all potential edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    m = len(all_edges)  # 190

    def edge_count(adj):
        return sum(len(s) for s in adj) // 2

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    def is_k23_free(adj):
        """Check if graph is K2,3-free: no two vertices have 3+ common neighbors"""
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    if len(adj[u] & adj[v]) >= 3:
                        return False
        return True

    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free"""
        if v in adj[u]:
            return False
            
        # Check: u and v would have too many common neighbors
        if len(adj[u] & adj[v]) >= 3:
            return False
            
        # For each neighbor w of u (not v), check if v would be 3rd common neighbor of u and w
        for w in adj[u]:
            if w == v:
                continue
            common_uw = adj[u] & adj[w]
            # v is not currently in adj[u], so v not in common_uw
            # After adding (u,v), v would be in common neighbors of u and w if w is neighbor of v
            if w in adj[v] and len(common_uw) >= 2:
                return False
                
        # For each neighbor w of v (not u), check if u would be 3rd common neighbor of v and w
        for w in adj[v]:
            if w == u:
                continue
            common_vw = adj[v] & adj[w]
            # u is not currently in adj[v], so u not in common_vw
            # After adding (u,v), u would be in common neighbors of v and w if w is neighbor of u
            if w in adj[u] and len(common_vw) >= 2:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(adj, u, v):
        adj[u].remove(v)
        adj[v].remove(u)

    # Multi-start local search with multiple random initializations
    best_edges = []
    best_count = 0
    
    # Run multiple restarts
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Reset
        adj = [set() for _ in range(n)]
        
        # Random greedy initialization
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if can_add(adj, u, v):
                add_edge(adj, u, v)
        
        current_count = edge_count(adj)
        
        # Local search: try to add missing edges, then 1-swap, then 2-swap
        for iteration in range(2000):
            if time.time() - start_time > timeout:
                break
            
            # Try to add any missing edge
            added = False
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            random.shuffle(missing)
            
            for u, v in missing:
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    added = True
            
            if not added:
                # Try 1-swap: remove edge e1, add edge e2
                all_current = get_edges(adj)
                random.shuffle(all_current)
                
                found_swap = False
                for e1 in all_current:
                    u1, v1 = e1
                    remove_edge(adj, u1, v1)
                    
                    # Try to add a missing edge
                    missing_after = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
                    random.shuffle(missing_after)
                    
                    best_add = None
                    for u2, v2 in missing_after:
                        if can_add(adj, u2, v2):
                            best_add = (u2, v2)
                            break
                    
                    if best_add:
                        add_edge(adj, best_add[0], best_add[1])
                        found_swap = True
                        break
                    else:
                        # Restore
                        add_edge(adj, u1, v1)
                
                if not found_swap:
                    # Try 2-swap: remove two edges, add two edges
                    all_current = get_edges(adj)
                    random.shuffle(all_current)
                    
                    found_2swap = False
                    for i, e1 in enumerate(all_current[:50]):  # Limit for speed
                        u1, v1 = e1
                        for j in range(i+1, min(i+10, len(all_current))):  # Limit pairs
                            e2 = all_current[j]
                            u2, v2 = e2
                            
                            remove_edge(adj, u1, v1)
                            remove_edge(adj, u2, v2)
                            
                            # Try to add two missing edges
                            missing_after = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
                            random.shuffle(missing_after)
                            
                            added_edges = []
                            for u3, v3 in missing_after:
                                if can_add(adj, u3, v3):
                                    add_edge(adj, u3, v3)
                                    added_edges.append((u3, v3))
                                    if len(added_edges) == 2:
                                        break
                            
                            if len(added_edges) == 2:
                                found_2swap = True
                                break
                            else:
                                # Restore
                                for ae in added_edges:
                                    remove_edge(adj, ae[0], ae[1])
                                add_edge(adj, u1, v1)
                                add_edge(adj, u2, v2)
                        
                        if found_2swap:
                            break
                    
                    if not found_2swap:
                        # Stuck, break
                        break
            
            current_count = edge_count(adj)
            
            if current_count > best_count:
                best_count = current_count
                best_edges = get_edges(adj)
    
    # Final verification and return
    if best_edges:
        # Verify the best solution is valid
        adj_check = [set() for _ in range(n)]
        for u, v in best_edges:
            add_edge(adj_check, u, v)
        if is_k23_free(adj_check):
            return best_edges
        else:
            # Fallback: find a valid subgraph
            adj_check = [set() for _ in range(n)]
            valid_edges = []
            for u, v in best_edges:
                if can_add(adj_check, u, v):
                    add_edge(adj_check, u, v)
                    valid_edges.append([u, v])
            return valid_edges
    else:
        # Fallback: return empty graph
        return []
