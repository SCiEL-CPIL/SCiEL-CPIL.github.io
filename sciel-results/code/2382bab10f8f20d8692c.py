import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Try to find a valid graph for n=39, then 38, etc.
    for n in range(39, 29, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random seeds
        # Number of attempts depends on remaining time
        # Each attempt: construction (~ms) + local search (~ms) + alpha check (~ms to s)
        # Let's cap attempts to avoid infinite loop if alpha check is slow
        max_attempts = 200 
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # 1. Construct random C4-free graph
            neighbors = [0] * n
            # List of all possible edges
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges_list)
            
            for u, v in edges_list:
                # Check C4 condition: N(u) and N(v) must be disjoint
                if (neighbors[u] & neighbors[v]) == 0:
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)
                    
            # 2. Local Search to reduce independence number
            # We want to add edges between vertices that are currently independent
            # to break up large independent sets.
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                
                improved = False
                # Find a maximal independent set S
                # Greedy: pick vertex with min degree in remaining, remove it and its neighbors
                remaining = (1 << n) - 1
                S = []
                while remaining:
                    # Pick vertex with minimum degree in 'remaining'
                    min_deg = n
                    pick = -1
                    temp = remaining
                    while temp:
                        v = (temp & -temp).bit_length() - 1
                        temp &= temp - 1
                        deg = (neighbors[v] & remaining).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            pick = v
                    S.append(pick)
                    remaining &= ~(neighbors[pick] | (1 << pick))
                
                # If we found an independent set of size > 10, try to fix it
                if len(S) > 10:
                    # Try to add edges between pairs in S
                    # If we can add an edge (u,v) in S without creating C4, do it
                    for i in range(len(S)):
                        u = S[i]
                        for j in range(i+1, len(S)):
                            v = S[j]
                            # Check if edge (u,v) can be added
                            if not (neighbors[u] >> v) & 1: # Not already an edge
                                if (neighbors[u] & neighbors[v]) == 0: # No C4
                                    neighbors[u] |= (1 << v)
                                    neighbors[v] |= (1 << u)
                                    improved = True
                                    # Break out of inner loop? 
                                    # We can continue to add more edges in this S if possible
                                    # But adding one edge might change degrees/C4 constraints for others in S?
                                    # Yes, N(u) changes, so (N(u) & N(w)) might no longer be 0 for some w in S.
                                    # So we should probably restart the S search or just continue carefully.
                                    # Given the low cost, continuing is fine, but we must re-check C4 condition for subsequent pairs.
                                    # The check (neighbors[u] & neighbors[v]) == 0 is done *before* adding, so it's safe for that specific pair.
                                    # However, if we add (u,v), it might prevent adding (u,w) later if w was a neighbor of v? 
                                    # No, w is in S, so w is NOT a neighbor of v in the current graph.
                                    # But wait, if we add (u,v), u and v become neighbors.
                                    # Does this affect the ability to add (u,w)?
                                    # Condition for (u,w): N(u) and N(w) disjoint.
                                    # N(u) now includes v. So we need v not in N(w).
                                    # Since w is in S (independent set), w is not adjacent to v in the *original* graph.
                                    # But we are modifying the graph.
                                    # If we add (u,v), u and v are adjacent.
                                    # If we then try to add (u,w), we check if N(u) & N(w) == 0.
                                    # N(u) contains v. So we need v not in N(w).
                                    # Is v in N(w)?
                                    # Initially, S was independent, so v not in N(w).
                                    # Did we add an edge between v and w?
                                    # We iterate i, j. If we added (v,w) earlier, then v is in N(w).
                                    # So the check `(neighbors[u] & neighbors[v]) == 0` handles this dynamically.
                                    pass

            # 3. Exact Check for Alpha <= 10
            alpha = exact_max_independent_set(neighbors, n, deadline)
            
            if alpha is not None and alpha <= 10:
                # Construct edge list
                edges = []
                for i in range(n):
                    mask = neighbors[i]
                    while mask:
                        j = (mask & -mask).bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask &= mask - 1
                return {'n': n, 'edges': edges}

    # Fallback: 30 vertices, 10 triangles
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def exact_max_independent_set(neighbors, n, deadline):
    """
    Returns the size of the maximum independent set, or None if timeout.
    """
    best_size = 0
    
    def bnb(vertices, current_size):
        nonlocal best_size
        if time.time() > deadline:
            return -1 # Timeout signal
        
        if not vertices:
            if current_size > best_size:
                best_size = current_size
            return 0
        
        # Pruning: even if we pick all remaining, we can't beat best
        if current_size + vertices.bit_count() <= best_size:
            return 0
            
        # Heuristic: Pick vertex with highest degree in remaining subgraph
        # This tends to prune the "include" branch faster (removes many neighbors)
        # and the "exclude" branch leaves fewer vertices.
        # Actually, for independent set, picking a high degree vertex to INCLUDE is very restrictive.
        # Picking a low degree vertex to INCLUDE is less restrictive.
        # Standard heuristic: Pick vertex with max degree in remaining graph.
        
        max_deg = -1
        pick = -1
        temp = vertices
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = (neighbors[v] & vertices).bit_count()
            if deg > max_deg:
                max_deg = deg
                pick = v
        
        # Branch 1: Include pick
        # New vertices: vertices without pick and without its neighbors
        new_vertices = vertices & ~(neighbors[pick] | (1 << pick))
        ret = bnb(new_vertices, current_size + 1)
        if ret == -1: return -1
        
        # Branch 2: Exclude pick
        new_vertices = vertices & ~(1 << pick)
        ret = bnb(new_vertices, current_size)
        if ret == -1: return -1
        
        return 0

    all_vertices = (1 << n) - 1
    ret = bnb(all_vertices, 0)
    if ret == -1:
        return None
    return best_size
