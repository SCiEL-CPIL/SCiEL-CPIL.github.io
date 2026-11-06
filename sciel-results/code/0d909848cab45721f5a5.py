import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We know n=26 is achievable via algebraic construction (Lazebnik-Woldar).
    # The seed program returns n=11.
    # The inspirations found n=26 (score 2.0) and n=26 (score 2.09).
    # To maximize the score (n/11), we should aim for the highest n possible.
    # n=26 is the known bound. n=27 or 28 would be new but unlikely to be found by heuristic search in 120s.
    # However, the inspirations used local search to *find* a valid coloring for n=26.
    # The algebraic construction is deterministic and instant.
    # Let's implement the algebraic construction for n=26 directly.
    # If that fails (it shouldn't), we fall back to local search.
    
    # Lazebnik-Woldar construction for 5-coloring K_26 without monochromatic C4.
    # Vertices are elements of Z_5 x Z_5.
    # n = 25? Wait, the task says "algebraic construction ... gives 26 vertices".
    # Let's check the standard construction.
    # R_5(C4) > 25 is known. R_5(C4) > 26 is known?
    # The prompt says: "The algebraic construction of Lazebnik and Woldar gives 26 vertices; 27 or 28 would be new."
    # So n=26 is the target.
    
    # Construction details for n=26:
    # One common construction uses vertices (x, y) in Z_5 x Z_5 plus one point at infinity?
    # Or is it simply 26 vertices?
    # Let's look at the inspirations. They used local search to find a valid coloring for n=26.
    # This implies a valid coloring exists and can be found.
    # Since I don't have the explicit algebraic formula memorized for n=26 specifically (often it's 25 for affine plane of order 5, which gives R_5(C4) > 25), 
    # and the prompt explicitly mentions 26, I will rely on the local search approach from the inspirations, 
    # but optimized to be faster and more robust.
    
    # Actually, let's try to construct n=26 using the local search method from inspiration 2, 
    # but with a tighter deadline and better initialization if possible.
    # The inspirations scored 2.0 and 2.09, which corresponds to n=22 and n=23? 
    # Score = n / 11.
    # 2.0 * 11 = 22.
    # 2.09 * 11 = 22.99 -> 23.
    # So the inspirations only found n=22 and n=23.
    # The prompt says "The algebraic construction ... gives 26 vertices". This implies n=26 is possible.
    # If I can implement the algebraic construction, I get score 26/11 = 2.36.
    # If I can't, I should try to beat 23.
    
    # Let's try to find the algebraic construction for n=26.
    # A known result is R_5(C4) > 25 (from affine plane of order 5).
    # Is there a construction for 26?
    # Yes, Lazebnik and Woldar (1995) proved R_5(C4) > 26.
    # The construction involves a specific graph.
    # Without the explicit formula, I will use a high-performance local search.
    # To improve upon the inspirations (which got 22-23), I need a better search.
    # The inspirations used simulated annealing.
    # A key improvement is using a greedy initialization or a more efficient violation calculation.
    
    # Let's implement a fast local search for n=26.
    # If it fails, try n=25, 24, etc.
    
    target_ns = [26, 25, 24, 23, 22]
    
    for n in target_ns:
        if time.time() > deadline - 10:
            break
        # Allocate remaining time to the search, with a safety margin
        time_left = deadline - time.time() - 5
        if time_left < 2:
            break
        result = local_search_fast(n, time_limit=time_left)
        if result is not None:
            return result
            
    # Fallback to seed
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def local_search_fast(n, time_limit=20):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge index for O(1) lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Spectral Initialization
    # Generate random symmetric matrix
    A = np.random.randn(n, n)
    A = (A + A.T) / 2.0
    
    # Get top 5 eigenvectors
    # For n=26, eigen decomposition is fast
    try:
        w, v = np.linalg.eigh(A)
        # eigh returns eigenvalues in ascending order
        # We want the top 5 (largest magnitude, or just the last 5)
        top_eigvecs = v[:, -5:]
    except:
        top_eigvecs = np.random.randn(n, 5)

    # Assign colors based on eigenvector components
    # Strategy: For each edge (i, j), compute a score vector s_k = v_k[i] * v_k[j] for k=0..4
    # Or use the sum of squared differences?
    # Let's use a quantile-based assignment on the "alignment" score.
    # Actually, a simpler spectral method:
    # Project vertices onto 5D space using top eigenvectors.
    # For edge (i,j), compute the angle or dot product.
    # Let's try: score_k = top_eigvecs[i, k] * top_eigvecs[j, k]
    # Then assign color based on which component is largest?
    # This might not be balanced.
    
    # Better: Compute a random linear combination of the top eigenvectors to get a single score per edge?
    # No, we need 5 colors.
    
    # Let's use the following:
    # For each vertex i, we have a 5D vector e_i.
    # For edge (i,j), compute dot product d = e_i . e_j.
    # This gives a single scalar. We need to map this to 5 colors.
    # This is effectively a 1D clustering.
    
    # Alternative: Use the 5 components as 5 separate scores.
    # color = argmax_k (e_i[k] * e_j[k])?
    # This is not guaranteed to be balanced.
    
    # Let's stick to a robust random initialization but perturbed by spectral structure?
    # Or just use the spectral scores to bias the initial random choice?
    
    # Let's try the argmax approach first, then rebalance if needed.
    colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        scores = top_eigvecs[i] * top_eigvecs[j]
        # Add small noise to break ties
        scores += np.random.randn(5) * 0.1
        colors[idx] = np.argmax(scores)

    # Build adjacency masks for each color
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        # Remove from old_color
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        # Add to new_color
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    # Calculate violations
    # A violation is a pair of vertices (u, v) and a color c such that they have >1 common neighbor in color c.
    # We count the number of C4s.
    def calc_violations():
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations
        
    current_violations = calc_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    # Precompute bitmasks for vertices for faster operations?
    # neighbors[c][v] is already an int bitmask.
    
    # Incremental violation update
    # When changing edge (i, j) from c1 to c2:
    # 1. In color c1, removing (i, j):
    #    - Pairs (i, u) where u is a neighbor of j in c1: common neighbors decrease by 1.
    #    - Pairs (j, u) where u is a neighbor of i in c1: common neighbors decrease by 1.
    #    - Pairs (u, v) where u is neighbor of i, v is neighbor of j? No, i and j are the endpoints.
    #    - Actually, the C4s involving edge (i, j) are those where (i, j) is part of the cycle.
    #    - A C4 is (u, i, v, j, u). The edges are (u,i), (i,v), (v,j), (j,u).
    #    - If we remove (i, j), it breaks C4s where (i, j) is an edge.
    #    - But our violation count is based on "common neighbors".
    #    - Violation for pair (u, v) in color c is C(k, 2) where k = |N_c(u) ∩ N_c(v)|.
    #    - Changing edge (i, j) affects:
    #      - For color c1:
    #        - Pair (i, j): N_c1(i) loses j, N_c1(j) loses i. Common neighbors of i and j?
    #          N_c1(i) ∩ N_c1(j). Since j is in N_c1(i) and i is in N_c1(j), they are in each other's sets?
    #          Wait, neighbors are distinct vertices. i is not in N(i).
    #          So j is in N(i), i is in N(j).
    #          Common neighbors of i and j are vertices w such that w~i and w~j.
    #          Removing (i, j) does not change N(i) ∩ N(j) except that j is removed from N(i) and i from N(j).
    #          Does j belong to N(i) ∩ N(j)? No, j is not in N(j).
    #          So the set of common neighbors of i and j is unchanged?
    #          Wait. If w is a common neighbor, w~i and w~j.
    #          The edge (i, j) removal doesn't affect w~i or w~j.
    #          So the violation count for pair (i, j) is unchanged?
    #          BUT, the pairs (i, u) and (j, u) are affected.
    #          For pair (i, u): N(i) loses j. So if j was in N(u), then j was a common neighbor of i and u.
    #          So k decreases by 1 if j ~_c u.
    #          Similarly for pair (j, u): N(j) loses i. k decreases by 1 if i ~_c u.
    #      - For color c2:
    #        - Adding (i, j).
    #        - Pair (i, j): No change in common neighbors (as above).
    #        - Pair (i, u): N(i) gains j. k increases by 1 if j ~_c2 u.
    #        - Pair (j, u): N(j) gains i. k increases by 1 if i ~_c2 u.
    
    # So we can update violations incrementally.
    # Let V(c, u, v) be the number of common neighbors of u and v in color c.
    # Violation contribution is V(V-1)/2.
    # Let delta_V be the change in V.
    # Change in contribution: (V+delta)(V+delta-1)/2 - V(V-1)/2
    
    def get_V(c, u, v):
        return (neighbors[c][u] & neighbors[c][v]).bit_count()
    
    def delta_cost(V, dV):
        if V + dV < 0: return 0 # Should not happen
        old = V * (V - 1) // 2
        new = (V + dV) * (V + dV - 1) // 2
        return new - old

    # Local search loop
    while time.time() < start_time + time_limit:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        i, j = edges[e_idx]
        
        # Calculate delta violations
        delta_total = 0
        
        # Effect on old_color
        # Pairs (i, u) for u in N_old(i) \ {j} ? No, for all u != i, j.
        # Actually, we iterate over neighbors.
        # For u in range(n):
        #   if u == i or u == j: continue
        #   if u is neighbor of i in old_color:
        #      V_old = get_V(old_color, i, u)
        #      # j is neighbor of i in old_color. Is j neighbor of u in old_color?
        #      if neighbors[old_color][u] & (1 << j):
        #          # j was a common neighbor of i and u
        #          delta_total += delta_cost(V_old, -1)
        #   if u is neighbor of j in old_color:
        #      V_old = get_V(old_color, j, u)
        #      if neighbors[old_color][u] & (1 << i):
        #          delta_total += delta_cost(V_old, -1)
        
        # Optimization: Only check u that are neighbors of i or j in old_color.
        # N_i_old = neighbors[old_color][i]
        # N_j_old = neighbors[old_color][j]
        # Candidates for u are in N_i_old | N_j_old.
        
        N_i_old = neighbors[old_color][i]
        N_j_old = neighbors[old_color][j]
        
        # Remove i, j from consideration (they are not their own neighbors)
        # N_i_old does not contain i. N_j_old does not contain j.
        # But N_i_old might contain j. N_j_old might contain i.
        
        # For each u in N_i_old:
        #   If u == j: skip (j is the other endpoint)
        #   V = get_V(old_color, i, u)
        #   if u ~_old j: delta += delta_cost(V, -1)
        
        # For each u in N_j_old:
        #   If u == i: skip
        #   V = get_V(old_color, j, u)
        #   if u ~_old i: delta += delta_cost(V, -1)
        
        # Note: If u is in both N_i_old and N_j_old, we count it twice?
        # No, the pairs are (i, u) and (j, u). They are distinct pairs.
        
        # Let's implement this carefully.
        
        # Effect on old_color (removal)
        # Iterate u in N_i_old
        temp_mask = N_i_old
        while temp_mask:
            lsb = temp_mask & -temp_mask
            u = lsb.bit_length() - 1
            temp_mask ^= lsb
            if u == j: continue
            # Check if u is neighbor of j in old_color
            if N_j_old & (1 << u):
                V = (N_i_old & neighbors[old_color][u]).bit_count()
                # Wait, N_i_old is N(i). We need N(i) ∩ N(u).
                # But we are removing j from N(i).
                # So new N(i) = N_i_old ^ (1<<j).
                # V_old = |N(i) ∩ N(u)|.
                # V_new = |N(i) \ {j} ∩ N(u)|.
                # If j in N(u), V_new = V_old - 1. Else V_new = V_old.
                # We already checked if j in N(u) (via N_j_old & (1<<u)).
                # So V_old is computed on current state.
                # We need V_old BEFORE update.
                V_old = (N_i_old & neighbors[old_color][u]).bit_count()
                delta_total += delta_cost(V_old, -1)
        
        # Iterate u in N_j_old
        temp_mask = N_j_old
        while temp_mask:
            lsb = temp_mask & -temp_mask
            u = lsb.bit_length() - 1
            temp_mask ^= lsb
            if u == i: continue
            if N_i_old & (1 << u):
                V_old = (N_j_old & neighbors[old_color][u]).bit_count()
                delta_total += delta_cost(V_old, -1)
                
        # Effect on new_color (addition)
        N_i_new = neighbors[new_color][i]
        N_j_new = neighbors[new_color][j]
        
        # Iterate u in N_i_new
        temp_mask = N_i_new
        while temp_mask:
            lsb = temp_mask & -temp_mask
            u = lsb.bit_length() - 1
            temp_mask ^= lsb
            if u == j: continue
            if N_j_new & (1 << u):
                V_old = (N_i_new & neighbors[new_color][u]).bit_count()
                delta_total += delta_cost(V_old, 1)
        
        # Iterate u in N_j_new
        temp_mask = N_j_new
        while temp_mask:
            lsb = temp_mask & -temp_mask
            u = lsb.bit_length() - 1
            temp_mask ^= lsb
            if u == i: continue
            if N_i_new & (1 << u):
                V_old = (N_j_new & neighbors[new_color][u]).bit_count()
                delta_total += delta_cost(V_old, 1)
                
        # Apply change
        colors[e_idx] = new_color
        update_neighbors(e_idx, old_color, new_color)
        current_violations += delta_total
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            if best_violations == 0:
                return {'n': n, 'colors': colors}
                
        # Simulated Annealing acceptance
        if delta_total >= 0:
            if random.random() >= pow(2.71828, -delta_total / temperature):
                # Revert
                colors[e_idx] = old_color
                update_neighbors(e_idx, new_color, old_color)
                current_violations -= delta_total
                
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None

