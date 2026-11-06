def local_search(adj, n, max_iters=500, deadline=None):
    """Local search: flip edges to eliminate regular induced subgraphs."""
    rng = random.Random()
    current_adj = adj[:]
    
    # Initial check
    violation = has_regular_induced_6plus(current_adj, n)
    if violation is None:
        return current_adj, True
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # violation is not None
        # Get vertices in violation
        verts = [i for i in range(n) if (violation >> i) & 1]
        
        # Try flipping a random edge within the violation
        if len(verts) >= 2:
            u, v = rng.sample(verts, 2)
        else:
            u, v = rng.sample(range(n), 2)
            
        # Flip edge
        if (current_adj[u] >> v) & 1:
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
        else:
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
        
        # Check if the specific violation is broken
        if not is_regular_subgraph_mask(violation, current_adj, n):
            # Violation is broken. Check if any other violation exists.
            # Full scan
            new_violation = has_regular_induced_6plus(current_adj, n)
            if new_violation is None:
                return current_adj, True
            else:
                violation = new_violation
        # If violation is still there, we keep trying to flip edges within it
        # (Next iteration will pick another edge in the same violation)
        
    # Final check
    violation = has_regular_induced_6plus(current_adj, n)
    return current_adj, (violation is None)
