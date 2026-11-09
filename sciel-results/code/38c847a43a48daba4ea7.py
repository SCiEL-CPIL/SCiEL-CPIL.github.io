import time
import random
import sys

def _alpha_bnb(adj, n):
    """
    Exact branch and bound to find the size of the maximum independent set.
    adj: list of bitmasks, adj[v] is the bitmask of neighbors of v.
    n: number of vertices.
    Returns the size of the maximum independent set.
    """
    best = [0]
    
    def bnb(cand, sz, depth):
        if sz + cand.bit_count() <= best[0]:
            return
        if cand == 0:
            if sz > best[0]:
                best[0] = sz
            return
        
        # Heuristic: pick vertex with max degree in cand
        # This helps prune the "include" branch faster
        best_v = -1
        best_deg = -1
        # Iterate over set bits in cand
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return

        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New candidates are neighbors of best_v excluded from cand
        bnb(cand & ~v_bit & ~adj[best_v], sz + 1, depth + 1)
        
        # Branch 2: Exclude best_v
        bnb(cand & ~v_bit, sz, depth + 1)

    bnb((1 << n) - 1, 0, 0)
    return best[0]

def _find_independent_set(adj, n, target_size):
    """
    Tries to find an independent set of size target_size.
    Returns the set of vertices if found, else None.
    """
    found = [None]
    
    def bnb(cand, sz, path):
        if found[0] is not None:
            return
        if sz >= target_size:
            found[0] = list(path)
            return
        if sz + cand.bit_count() < target_size:
            return
        if cand == 0:
            return
        
        best_v = -1
        best_deg = -1
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb
            
        if best_v == -1:
            return

        v_bit = 1 << best_v
        
        # Try include
        path.append(best_v)
        bnb(cand & ~v_bit & ~adj[best_v], sz + 1, path)
        path.pop()
        
        # Try exclude
        bnb(cand & ~v_bit, sz, path)

    bnb((1 << n) - 1, 0, [])
    return found[0]

def _is_triangle_free(adj, n):
    """
    Checks if the graph defined by adj is triangle-free.
    """
    for u in range(n):
        # Check if any two neighbors of u are connected
        # (adj[u] & adj[v]) == 0 for all v in adj[u]
        # More efficient: for each u, check if adj[u] & (union of adj[v] for v in adj[u]) has any edges?
        # Simpler: for each edge (u,v), check if (adj[u] & adj[v]) != 0.
        # But we can just iterate edges.
        pass
    
    # Efficient check:
    for u in range(n):
        neighbors = adj[u]
        # For each pair of neighbors v, w, check if v~w
        # This is O(n * deg^2).
        # Alternative: 
        # A triangle exists if there is an edge (u,v) such that adj[u] & adj[v] != 0.
        # We can iterate over edges.
        pass
        
    # Let's do the edge iteration, it's O(E * avg_deg) roughly.
    for u in range(n):
        v_bits = adj[u]
        while v_bits:
            lsb = v_bits & -v_bits
            v = lsb.bit_length() - 1
            if u < v: # Only check each edge once
                if (adj[u] & adj[v]) != 0:
                    return False
            v_bits ^= lsb
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Fallback: Andrásfai graph And(3) on 26 vertices
    n_fallback = 26
    edges_fallback = [[i, j] for i in range(n_fallback) for j in range(i + 1, n_fallback) if (j - i) % n_fallback % 3 == 1]
    
    # Try to find a graph on n=39
    n_target = 39
    
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        # Generate a random triangle-free graph
        # Start with empty graph
        adj = [0] * n_target
        
        # Add edges randomly, ensuring no triangle
        # Iterate over potential edges in random order
        edges_to_consider = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        random.shuffle(edges_to_consider)
        
        for u, v in edges_to_consider:
            # Check if adding (u,v) creates a triangle
            # Triangle exists if there is w such that (u,w) and (v,w) are edges.
            # i.e., adj[u] & adj[v] != 0
            if (adj[u] & adj[v]) == 0:
                # Safe to add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Optional: stop if graph becomes too dense? 
            # Or just let it be random.
        
        # Check triangle-free (should be by construction, but verify)
        if not _is_triangle_free(adj, n_target):
            continue
            
        # Check independence number
        alpha = _alpha_bnb(adj, n_target)
        
        if alpha < 10:
            # Convert adj to edges list
            edges = []
            for u in range(n_target):
                v_bits = adj[u]
                while v_bits:
                    lsb = v_bits & -v_bits
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    v_bits ^= lsb
            return {'n': n_target, 'edges': edges}
        
        # If alpha >= 10, try to repair
        # Find an independent set of size 10
        iso_set = _find_independent_set(adj, n_target, 10)
        if iso_set:
            # Try to add edges within iso_set to break it
            # We need to add an edge (u,v) with u,v in iso_set such that no triangle is formed.
            # Since u,v are non-adjacent, they share no common neighbor in the current graph?
            # No, they are independent, so no edge between them.
            # Adding (u,v) creates a triangle if there is w such that (u,w) and (v,w) are edges.
            # i.e., adj[u] & adj[v] != 0.
            
            modified = False
            # Shuffle iso_set to try different pairs
            random.shuffle(iso_set)
            for i in range(len(iso_set)):
                for j in range(i + 1, len(iso_set)):
                    u, v = iso_set[i], iso_set[j]
                    if (adj[u] & adj[v]) == 0:
                        # Can add edge (u,v)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        modified = True
                        break
                if modified:
                    break
            
            if modified:
                # Re-check alpha
                # Note: Adding an edge can only decrease or keep alpha same.
                # It cannot increase alpha.
                # So if we broke one independent set, alpha might still be >= 10 via another set.
                # We can loop, but for simplicity, we just check once more.
                alpha_new = _alpha_bnb(adj, n_target)
                if alpha_new < 10:
                    edges = []
                    for u in range(n_target):
                        v_bits = adj[u]
                        while v_bits:
                            lsb = v_bits & -v_bits
                            v = lsb.bit_length() - 1
                            if u < v:
                                edges.append([u, v])
                            v_bits ^= lsb
                    return {'n': n_target, 'edges': edges}
    
    # If no better graph found, return fallback
    return {'n': n_fallback, 'edges': edges_fallback}
