import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def find_violation(n, adj):
        """
        Returns a bitmask of a regular induced subgraph of size >= 6, or None if none exists.
        """
        # Iterate over all masks
        # Optimization: We can iterate masks in a random order to find violations faster on average?
        # No, sequential is fine.
        
        # Pre-computed list of masks with popcount >= 6 for this n?
        # Generating it takes time. For n=20, 1M masks.
        # Just loop and filter.
        
        for mask in range(1, 1 << n):
            # Quick filter
            if mask.bit_count() < 6:
                continue
            
            # Extract vertices
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Check regularity
            # All vertices must have the same degree within the mask
            d0 = (adj[verts[0]] & mask).bit_count()
            is_regular = True
            for v in verts[1:]:
                if (adj[v] & mask).bit_count() != d0:
                    is_regular = False
                    break
            
            if is_regular:
                return mask
        
        return None

    def get_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def try_find(n, time_limit):
        """
        Attempts to find a valid graph on n vertices using local search.
        """
        # Multiple restarts
        while time.time() < time_limit:
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            # To avoid getting stuck, we can limit the number of steps per restart
            # or just run until time limit.
            
            steps = 0
            max_steps = 5000 # Arbitrary, to allow restarts
            
            while time.time() < time_limit and steps < max_steps:
                steps += 1
                viol_mask = find_violation(n, adj)
                
                if viol_mask is None:
                    # Valid graph found
                    return {'n': n, 'edges': get_edges(n, adj)}
                
                # Get vertices in violation
                verts = []
                m = viol_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m ^= lsb
                
                # Flip a random edge within the violating subset
                # This is likely to break the regularity
                if len(verts) >= 2:
                    u_idx = random.randint(0, len(verts) - 1)
                    v_idx = random.randint(0, len(verts) - 1)
                    if u_idx != v_idx:
                        u = verts[u_idx]
                        v = verts[v_idx]
                        
                        # Flip edge
                        if adj[u] & (1 << v):
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    else:
                        # If we picked same vertex (shouldn't happen with distinct indices from sample, 
                        # but randint allows same), pick another
                        # Actually, just break if this happens? No, just continue loop
                        pass
                else:
                    # Should not happen for size >= 6
                    break
            
            # If we exhausted steps without success, restart with new random graph
            # The outer while loop handles this
            
        return None

    # Strategy:
    # 1. Try n=20 for 80s.
    # 2. If found, try n=21 for remaining time.
    # 3. If n=20 not found, try n=19, 18...
    
    best_graph = None
    
    # Try n=20
    t_limit_20 = start_time + 80
    g20 = try_find(20, t_limit_20)
    if g20:
        best_graph = g20
        
    # Try n=21 if we have time and a base graph
    if best_graph and time.time() < deadline - 10:
        # We can try to extend the 20-graph to 21
        # But try_find starts from scratch.
        # Extending is smarter.
        
        # Reconstruct adj from best_graph
        n_base = best_graph['n']
        adj = [0] * n_base
        for e in best_graph['edges']:
            u, v = e
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Add vertex n_base
        adj.append(0)
        # Connect randomly
        for i in range(n_base):
            if random.random() < 0.5:
                adj[i] |= (1 << n_base)
                adj[n_base] |= (1 << i)
        
        # Local search on the extended graph
        # We can reuse the logic from try_find but on this specific adj
        # Let's write a helper or just inline it.
        
        n_new = n_base + 1
        t_limit_21 = deadline - 5
        
        steps = 0
        max_steps = 2000
        while time.time() < t_limit_21 and steps < max_steps:
            steps += 1
            viol_mask = find_violation(n_new, adj)
            
            if viol_mask is None:
                # Valid 21-graph found!
                return {'n': n_new, 'edges': get_edges(n_new, adj)}
            
            # Fix violation
            verts = []
            m = viol_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            if len(verts) >= 2:
                u_idx = random.randint(0, len(verts) - 1)
                v_idx = random.randint(0, len(verts) - 1)
                if u_idx != v_idx:
                    u = verts[u_idx]
                    v = verts[v_idx]
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
    # If n=20 failed, try smaller
    if best_graph is None:
        for n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
            if time.time() > deadline - 5:
                break
            t_limit = min(deadline - 5, start_time + 20) # Give 20s to each
            g = try_find(n, t_limit)
            if g:
                best_graph = g
                break
    
    if best_graph is None:
        # Fallback
        return {'n': 5, 'edges': []}
        
    return best_graph
