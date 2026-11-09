import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 256
    full_mask = (1 << N) - 1
    
    # Build conflict graph
    # conflict[i] is a bitmask of j such that i and j conflict (0-indexed i, j)
    # Value v corresponds to index v-1.
    conflict = [0] * N
    
    for a_val in range(1, N + 1):
        a_idx = a_val - 1
        a_sq = a_val * a_val
        limit = int(a_val ** 0.5)
        for k in range(1, limit + 1):
            if a_sq % k == 0:
                d1 = k
                d2 = a_sq // k
                for d in (d1, d2):
                    if a_val < d <= a_val + N:
                        b_val = d - a_val
                        if 1 <= b_val <= N and b_val != a_val:
                            b_idx = b_val - 1
                            conflict[a_idx] |= (1 << b_idx)
                            conflict[b_idx] |= (1 << a_idx)
    
    # Build complement graph for Max Clique
    # comp_adj[i] has bit j set if i and j do NOT conflict and i != j
    comp_adj = [0] * N
    for i in range(N):
        # All bits except i and conflict[i]
        comp_adj[i] = (full_mask ^ (1 << i)) ^ conflict[i]
    
    best_size = 0
    best_clique = 0  # bitmask of the best clique found
    
    # Helper to update best
    def update_best(clique_mask, size):
        nonlocal best_size, best_clique
        if size > best_size:
            best_size = size
            best_clique = clique_mask

    # Greedy coloring to get upper bound on clique size in subgraph P
    def upper_bound(P):
        if P == 0:
            return 0
        colors = 0
        uncolored = P
        while uncolored:
            colors += 1
            # Find a maximal independent set in the subgraph induced by uncolored
            # This is equivalent to a greedy coloring step
            candidates = uncolored
            color_class = 0
            while candidates:
                # Pick a vertex from candidates
                # Heuristic: Pick vertex with highest degree in uncolored? 
                # For speed, just pick the lowest set bit or any.
                # Let's pick the one with highest degree in 'uncolored' to tighten bound?
                # Actually, standard Tomita uses a specific ordering.
                # For simplicity and speed, let's just pick the first one.
                v_bit = candidates & -candidates
                v_idx = v_bit.bit_length() - 1
                
                color_class |= v_bit
                uncolored ^= v_bit
                candidates ^= v_bit
                # Remove neighbors of v from candidates (they can't share this color)
                candidates &= ~comp_adj[v_idx]
        return colors

    # Max Clique Branch and Bound
    # P: candidates to add
    # X: candidates to exclude (for ordering)
    # depth: current clique size
    # clique_mask: current clique bitmask
    def expand(P, X, depth, clique_mask):
        nonlocal best_size, best_clique
        if time.time() > deadline:
            return
        
        if P == 0:
            if depth > best_size:
                best_size = depth
                best_clique = clique_mask
            return
        
        # Prune if upper bound doesn't beat best
        ub = upper_bound(P)
        if depth + ub <= best_size:
            return
            
        # Process vertices in P
        # We iterate through P, moving vertices to X as we go
        while P:
            if time.time() > deadline:
                return
            
            # Select a vertex v from P
            # Heuristic: Choose vertex with highest degree in P
            # To avoid O(N^2) overhead, we can just pick the lowest bit if N is small enough
            # But for 256, finding max degree is O(N) which is 256 ops.
            # Let's just pick the lowest set bit for simplicity first, 
            # but Tomita's algorithm benefits from ordering.
            # Let's use a simple heuristic: pick v with max degree in P.
            
            # Optimization: If we just pick any v, it might be slower.
            # Let's try to pick v with max degree in P.
            max_deg = -1
            v_idx = -1
            temp_P = P
            # Iterate through bits in P
            while temp_P:
                bit = temp_P & -temp_P
                idx = bit.bit_length() - 1
                deg = (comp_adj[idx] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v_idx = idx
                    v_bit = bit
                temp_P ^= bit
            
            if v_idx == -1:
                break

            # Remove v from P
            P ^= v_bit
            
            # Branch: Include v
            new_P = P & comp_adj[v_idx]
            new_clique_mask = clique_mask | v_bit
            expand(new_P, X & comp_adj[v_idx], depth + 1, new_clique_mask)
            
            # Move v to X (exclude v in subsequent iterations)
            X |= v_bit
            
            # Prune check after removing v
            ub = upper_bound(P)
            if depth + ub <= best_size:
                return

    # Initial call
    # P is all vertices, X is empty
    initial_P = full_mask
    initial_X = 0
    expand(initial_P, initial_X, 0, 0)
    
    # Convert best_clique bitmask to list of values
    result = []
    for i in range(N):
        if (best_clique >> i) & 1:
            result.append(i + 1)
            
    # If no result found (should not happen, at least 1 vertex), return [1]
    if not result:
        result = [1]
        
    return result
