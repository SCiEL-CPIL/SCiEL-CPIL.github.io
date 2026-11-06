<<<<<<< SEARCH
def local_search(adj_masks, n, rng, deadline):
    """
    Attempts to fix a graph by flipping edges in violating subgraphs.
    Returns (adj_masks, edges, is_valid)
    """
    cur_masks = adj_masks[:]
    
    # Initial check
    viol = find_violation_fast(cur_masks, n)
    if viol is None:
        return cur_masks, get_edges_from_masks(cur_masks, n), True
    
    max_iters = 500
    it = 0
    
    while it < max_iters:
        if time.time() > deadline:
            break
            
        viol = find_violation_fast(cur_masks, n)
        if viol is None:
            return cur_masks, get_edges_from_masks(cur_masks, n), True
            
        # Get the violating set
        vset = set(viol)
        
        # Candidate edges to flip:
        # 1. Edges inside the violating set (to break regularity)
        # 2. Non-edges inside the violating set (to break regularity)
        
        internal_edges = []
        internal_nonedges = []
        
        for u in vset:
            for v in vset:
                if u < v:
                    if cur_masks[u] & (1 << v):
                        internal_edges.append((u, v))
                    else:
                        internal_nonedges.append((u, v))
        
        # Strategy: Flip an edge within the violating set.
        # Removing an edge reduces degree of 2 vertices.
        # Adding an edge increases degree of 2 vertices.
        # This disrupts the equality of degrees.
        
        candidates = []
        if internal_edges:
            candidates.extend(internal_edges)
        if internal_nonedges:
            candidates.extend(internal_nonedges)
            
        if not candidates:
            # Should not happen for k>=6, but fallback to random
            u = rng.randint(0, n-1)
            v = rng.randint(0, n-1)
            if u == v:
                v = (v+1) % n
            candidates = [(u, v)]
            
        u, v = rng.choice(candidates)
        
        # Flip the edge
        cur_masks[u] ^= (1 << v)
        cur_masks[v] ^= (1 << u)
        
        it += 1
        
    # Final check
    viol = find_violation_fast(cur_masks, n)
    return cur_masks, get_edges_from_masks(cur_masks, n), viol is None
=======
def local_search(adj_masks, n, rng, deadline):
    """
    Attempts to fix a graph by flipping edges in violating subgraphs.
    Returns (adj_masks, edges, is_valid)
    """
    cur_masks = adj_masks[:]
    
    # Initial check
    viol = find_violation_fast(cur_masks, n)
    if viol is None:
        return cur_masks, get_edges_from_masks(cur_masks, n), True
    
    max_iters = 2000
    it = 0
    check_interval = 10
    
    while it < max_iters:
        if time.time() > deadline:
            break
            
        # Only check for violation every 'check_interval' steps to save time
        if it % check_interval == 0:
            viol = find_violation_fast(cur_masks, n)
            if viol is None:
                return cur_masks, get_edges_from_masks(cur_masks, n), True
        else:
            # If we didn't check, we don't have a current viol.
            # We can just pick a random edge to flip?
            # Or use the last known viol?
            # Let's use the last known viol if available.
            if 'viol' not in locals() or viol is None:
                viol = find_violation_fast(cur_masks, n)
                if viol is None:
                    return cur_masks, get_edges_from_masks(cur_masks, n), True

        vset = list(viol)
        
        # Flip a random edge inside the violating set
        u, v = rng.sample(vset, 2)
        
        # Flip the edge
        cur_masks[u] ^= (1 << v)
        cur_masks[v] ^= (1 << u)
        
        it += 1
        
    # Final check
    viol = find_violation_fast(cur_masks, n)
    return cur_masks, get_edges_from_masks(cur_masks, n), viol is None
>>>>>>> REPLACE
