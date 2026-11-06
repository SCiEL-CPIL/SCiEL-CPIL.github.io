import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 20
    num_colors = 20
    
    # Precompute all pairs
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_c4_free(adj):
        """
        Check if the graph defined by adj (list of int bitmasks) is C4-free.
        adj[u] has bit v set if (u,v) is an edge.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                # Check if adding (u,v) would have created a C4, 
                # or just check if a C4 exists in the current graph.
                # A C4 exists if there is a path of length 3 between u and v 
                # in the graph *without* the edge (u,v)? 
                # Actually, we are checking the current graph.
                # If (u,v) is in the graph, does it participate in a C4?
                # A C4 containing (u,v) is u-v-x-w-u.
                # This requires x in N(v) and w in N(u) and (x,w) in E.
                # So we check if N(v) and N(u) have a common neighbor? No, that's a triangle.
                # We need x in N(v) and w in N(u) such that (x,w) is an edge.
                
                # Let's just check all 4-cycles in the graph.
                pass # This function is not used directly in the greedy step, 
                     # we use the incremental check.
        
        return True

    def can_add_edge_c4_free(u, v, adj):
        """
        Check if adding edge (u,v) to the graph defined by adj creates a 4-cycle.
        adj[u] is bitmask of neighbors of u.
        """
        # If (u,v) is already there, return False (or True depending on definition, but we only call if not there)
        if (adj[u] >> v) & 1:
            return False
        
        # A 4-cycle u-v-x-w-u is created if there exist x, w such that:
        # 1. (v,x) is in adj
        # 2. (w,u) is in adj
        # 3. (x,w) is in adj
        # 4. x != u, w != v, x != w (simple cycle)
        
        # Iterate x in neighbors of v
        x_mask = adj[v]
        # Iterate w in neighbors of u
        w_mask = adj[u]
        
        # We need x in x_mask, w in w_mask, and (x,w) in adj.
        # Also x != u and w != v.
        
        # Optimize:
        # For each x in N(v):
        #   For each w in N(u):
        #     if x == u or w == v: continue
        #     if x == w: continue
        #     if (adj[x] >> w) & 1: return False
        
        # Better:
        # Let S = { (x,w) | x in N(v), w in N(u) }
        # Check if any (x,w) in S is an edge.
        
        # Since N is small, we can just loop.
        x_list = [i for i in range(n) if (x_mask >> i) & 1]
        w_list = [i for i in range(n) if (w_mask >> i) & 1]
        
        for x in x_list:
            if x == u:
                continue
            for w in w_list:
                if w == v or w == x:
                    continue
                if (adj[x] >> w) & 1:
                    return False
        return True

    def greedy_c4_free():
        adj = [0] * n
        edges = 0
        
        # Random order of edges
        shuffled_edges = list(all_edges)
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if can_add_edge_c4_free(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges += 1
                
        return adj, edges

    def local_search_c4_free(adj):
        # Try to add missing edges
        # If we can't add any, try to remove an edge and add two others? 
        # For now, just greedy is often strong for N=20.
        # Let's try a simple hill climbing: 
        # Remove a random edge, try to add all missing edges greedily again?
        # This is expensive. 
        # Instead, try to swap: remove e1, add e2.
        # If e2 is a missing edge that becomes addable after removing e1.
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Get list of current edges
            current_edges = set()
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        current_edges.add((u,v))
            
            missing_edges = [e for e in all_edges if e not in current_edges]
            
            # Try to add a missing edge
            for e in missing_edges:
                u, v = e
                if can_add_edge_c4_free(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add one
            if current_edges:
                e_remove = random.choice(list(current_edges))
                u_r, v_r = e_remove
                adj[u_r] &= ~(1 << v_r)
                adj[v_r] &= ~(1 << u_r)
                
                # Try to add any missing edge
                added = False
                for e_add in missing_edges:
                    u_a, v_a = e_add
                    if can_add_edge_c4_free(u_a, v_a, adj):
                        adj[u_a] |= (1 << v_a)
                        adj[v_a] |= (1 << u_a)
                        added = True
                        break
                
                if added:
                    improved = True
                else:
                    # Revert
                    adj[u_r] |= (1 << v_r)
                    adj[v_r] |= (1 << u_r)
        
        # Count edges
        edges = 0
        for u in range(n):
            edges += bin(adj[u]).count('1')
        edges //= 2
        return adj, edges

    best_adj = None
    best_edges = 0
    
    # Run multiple restarts
    while time.time() < deadline - 5:
        adj, edges = greedy_c4_free()
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
        # Local search on the best found so far? Or on this one?
        # Let's do local search on this one if it's promising
        adj_ls, edges_ls = local_search_c4_free(adj)
        if edges_ls > best_edges:
            best_edges = edges_ls
            best_adj = adj_ls[:]
            
    # Construct result
    # Color edges with 0..19
    result = []
    color_idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v, color_idx % num_colors])
                color_idx += 1
                
    return result
