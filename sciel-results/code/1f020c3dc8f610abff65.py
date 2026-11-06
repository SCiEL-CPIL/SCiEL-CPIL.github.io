import time
import random
import itertools

_START_TIME = None

def _get_time_left(deadline):
    if _START_TIME is None:
        return deadline
    return deadline - (time.time() - _START_TIME)

def _is_k4_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
    # adj is a list of bitmasks (ints)
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            common = adj[a] & adj[b]
            # Remove a and b
            common &= ~(1 << a)
            common &= ~(1 << b)
            # Check if common contains an edge
            # Iterate through bits in common
            c = common
            while c:
                # Get lowest set bit
                lsb = c & -c
                c ^= lsb
                idx_c = lsb.bit_length() - 1
                # Check if any other bit in common is connected to idx_c
                # common & adj[idx_c] should have a bit set
                if (common & adj[idx_c]):
                    return False
    return True

def _max_triangle_free_induced(adj, n, deadline, start_time):
    """Find largest triangle-free induced subgraph size via branch and bound."""
    best = 0
    deadline_time = start_time + deadline
    
    # Precompute triangles?
    # Or just check on the fly.
    
    def can_add(v, current_mask):
        # Check if v can be added to current_mask without forming a triangle
        # A triangle is formed if there are u, w in current_mask such that
        # adj[v][u], adj[v][w], adj[u][w] are all true.
        # Let neighbors of v in current_mask be N_v = adj[v] & current_mask
        # We need to check if N_v contains an edge.
        # i.e., is there u in N_v such that (adj[u] & N_v) != 0?
        # Actually, if N_v contains an edge (u,w), then v,u,w form a triangle.
        n_v = adj[v] & current_mask
        # Check if n_v is an independent set in the induced subgraph
        # If n_v has an edge, return False
        c = n_v
        while c:
            lsb = c & -c
            c ^= lsb
            idx = lsb.bit_length() - 1
            if (n_v & adj[idx]):
                return False
        return True

    def dfs(available_mask, current_mask, current_size):
        nonlocal best
        if time.time() > deadline_time:
            return
        
        # Update best
        if current_size > best:
            best = current_size
            
        if available_mask == 0:
            return
            
        # Upper bound
        if current_size + available_mask.bit_count() <= best:
            return
            
        # Heuristic: try to add vertices that are "safe"
        # Order vertices by degree in available?
        
        # Iterate through available vertices
        c = available_mask
        while c:
            lsb = c & -c
            c ^= lsb
            v = lsb.bit_length() - 1
            
            if can_add(v, current_mask):
                dfs(available_mask ^ lsb, current_mask | lsb, current_size + 1)
            # Option to skip v is implicit in the loop structure?
            # No, this DFS tries to add v.
            # We need to also consider NOT adding v.
            # Standard B&B for Max Independent Set:
            # Branch 1: Include v
            # Branch 2: Exclude v
            
            # However, for Triangle-Free, it's not a simple independent set.
            # The condition is global.
            
            # Let's stick to the recursive structure:
            # Pick a vertex v from available.
            # Branch 1: Include v (if valid).
            # Branch 2: Exclude v.
            
            # To avoid redundancy, we fix an ordering.
            # But the "valid" check depends on the whole current set.
            
            # Let's refine the DFS:
            # Pick the first available vertex v.
            # Branch 1: Add v (if can_add).
            # Branch 2: Remove v from available (exclude v).
            
            # This is better.
            break # We only pick the first one for the split
            
        # Re-implement DFS with fixed split
        
        if available_mask == 0:
            return
            
        # Pick lowest bit
        lsb = available_mask & -available_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        if can_add(v, current_mask):
            dfs(available_mask ^ lsb, current_mask | lsb, current_size + 1)
            
        # Branch 2: Exclude v
        dfs(available_mask ^ lsb, current_mask, current_size)

    dfs((1 << n) - 1, 0, 0)
    return best

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = 110
    n = 18
    
    best_edges = None
    best_score = 999
    
    def get_adjacency(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Start with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    
    adj = get_adjacency(edges, n)
    
    # Local Search
    # Try to improve score by flipping edges
    
    def evaluate(adj):
        if not _is_k4_free(adj, n):
            return 999
        return _max_triangle_free_induced(adj, n, 5, _START_TIME)
    
    current_score = evaluate(adj)
    best_score = current_score
    best_adj = adj[:]
    
    improved = True
    while improved and time.time() - _START_TIME < deadline - 10:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if time.time() - _START_TIME > deadline - 10:
                    break
                
                # Try flipping edge (u,v)
                if (adj[u] >> v) & 1:
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # Add edge
                    # Check K4 first for speed?
                    # Adding edge might create K4.
                    # Check if adding (u,v) creates a K4.
                    # A K4 containing (u,v) must have two other vertices x,y
                    # such that u,v,x,y form a K4.
                    # x,y must be common neighbors of u and v.
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    
                    # Check if common contains an edge (x,y)
                    # If yes, adding (u,v) creates K4 u,v,x,y.
                    c = common
                    creates_k4 = False
                    while c:
                        lsb = c & -c
                        c ^= lsb
                        idx_x = lsb.bit_length() - 1
                        if (common & adj[idx_x]):
                            creates_k4 = True
                            break
                    
                    if creates_k4:
                        # Revert
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        continue
                    
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_score = evaluate(adj)
                
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
                    improved = True
                else:
                    # Revert
                    if (adj[u] >> v) & 1:
                        # It was added, so remove
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        # It was removed, so add back
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
    
    if best_adj is None:
        best_adj = adj
        
    return adj_to_edges(best_adj, n)

construct()
