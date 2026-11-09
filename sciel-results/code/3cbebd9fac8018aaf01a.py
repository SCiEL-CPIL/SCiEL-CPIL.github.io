def construct():
    import time
    from math import gcd, lcm
    import sys

    start_time = time.time()
    deadline = start_time + 50

    N = 255
    values = list(range(2, 257))
    weights = [1.0 / v for v in values]
    
    # Build adjacency bitmasks for compatibility graph
    # adj_mask[i] has bit j set if values[i] and values[j] are compatible (lcm > 256)
    adj_mask = [0] * N
    for i in range(N):
        vi = values[i]
        mask = 0
        for j in range(i + 1, N):
            vj = values[j]
            if lcm(vi, vj) > 256:
                mask |= (1 << j)
        # Also set self? No, in BK, we don't need self in mask for intersection logic usually, 
        # but P contains the node itself? 
        # Standard BK: P is set of vertices.
        # If v is in P, and we recurse on v, new P = P ∩ N(v).
        # v is not in N(v) usually. So v is removed from P.
        # We need to ensure we don't pick v again.
        # In bitset BK:
        # candidates = P & ~N(pivot)
        # for v in candidates:
        #    ...
        #    P = P & N(v)  # This removes v if v not in N(v)
        #    X = X | (1 << v)
        # So we don't need self-loop in mask.
        
        adj_mask[i] = mask
        # Symmetry
        # We need adj_mask[j] to have bit i set.
        # Since we iterate i < j, we can do this in a second pass or handle carefully.
        # Easier: build full matrix or do two passes.
        
    # Re-build properly for symmetry
    adj_mask = [0] * N
    for i in range(N):
        vi = values[i]
        for j in range(i + 1, N):
            vj = values[j]
            if lcm(vi, vj) > 256:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    best_sum = 0.0
    best_clique_mask = 0
    
    # Helper to get list of indices from mask
    def mask_to_list(mask):
        res = []
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            res.append(idx)
            mask ^= lsb
        return res

    # Helper to sum weights of mask
    def sum_weights(mask):
        s = 0.0
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            s += weights[idx]
            mask ^= lsb
        return s

    # Greedy initial solution
    # Pick highest weight, remove non-neighbors
    P_work = (1 << N) - 1
    R_greedy_mask = 0
    sum_greedy = 0.0
    # To pick max weight, we might need to iterate. 
    # Since N is small, we can just iterate indices 0..N-1 and pick max weight available.
    available = [True] * N
    for _ in range(N):
        best_idx = -1
        best_w = -1.0
        for i in range(N):
            if available[i] and weights[i] > best_w:
                best_w = weights[i]
                best_idx = i
        if best_idx == -1:
            break
        R_greedy_mask |= (1 << best_idx)
        sum_greedy += best_w
        # Remove best_idx and its non-neighbors (conflicts)
        # Keep neighbors in compatibility graph
        # Non-neighbors are those i where adj_mask[best_idx] doesn't have bit i.
        # So we update available:
        # For each i, if i is not neighbor of best_idx and i != best_idx, available[i] = False
        # Actually, we want to keep only neighbors of best_idx (and best_idx itself is removed).
        # So new available = { i | i is neighbor of best_idx }
        # But we also need to remove best_idx from future consideration.
        # Let's just clear available for non-neighbors.
        for i in range(N):
            if i == best_idx:
                available[i] = False
            else:
                if not (adj_mask[best_idx] & (1 << i)):
                    available[i] = False
                    
    best_sum = sum_greedy
    best_clique_mask = R_greedy_mask

    def bk(R_mask, P_mask, X_mask, current_sum):
        nonlocal best_sum, best_clique_mask
        
        if not P_mask:
            if current_sum > best_sum:
                best_sum = current_sum
                best_clique_mask = R_mask
            return
            
        # Bound: Sum of weights in P
        # This is a valid upper bound.
        # We can make it tighter with coloring, but let's try this first with bitsets.
        # If P is large, sum_weights is O(N).
        bound_val = sum_weights(P_mask)
        
        if current_sum + bound_val <= best_sum:
            return
            
        # Pivot selection
        # Choose u in P U X maximizing popcount(P & adj_mask[u])
        # We only need to check u in P? Standard BK uses P U X.
        # Using P is a heuristic variant. Let's use P for speed.
        # Or better, iterate through bits in P and find max degree in P.
        
        max_deg = -1
        pivot = -1
        temp_P = P_mask
        while temp_P:
            lsb = temp_P & -temp_P
            u = lsb.bit_length() - 1
            # Count bits in P & adj_mask[u]
            # popcount is fast in Python 3.10+ via int.bit_count()
            deg = (P_mask & adj_mask[u]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = u
            temp_P ^= lsb
            
        if pivot == -1:
            return
            
        # Candidates: P \ N(pivot)
        # Note: N(pivot) in compatibility graph.
        # We want vertices in P that are NOT adjacent to pivot.
        # If they are not adjacent, they can be in the clique together with pivot? 
        # No, if we pick v, we require v to be adjacent to all in R.
        # The pivot logic: We only need to consider v in P \ N(pivot).
        # Because if v is in N(pivot), then any clique containing v and not pivot is also considered when pivot is processed?
        # Standard proof: Every maximal clique contains at least one vertex from P \ N(pivot).
        
        candidates = P_mask & ~adj_mask[pivot]
        
        # Iterate through candidates
        temp_C = candidates
        while temp_C:
            lsb = temp_C & -temp_C
            v = lsb.bit_length() - 1
            
            # Recurse
            new_R = R_mask | lsb
            new_P = P_mask & adj_mask[v]
            new_X = X_mask & adj_mask[v]
            
            bk(new_R, new_P, new_X, current_sum + weights[v])
            
            # Update P and X
            P_mask &= ~lsb
            X_mask |= lsb
            
            # Time check
            if time.time() > deadline:
                return

            temp_C ^= lsb

    # Initial call
    # P = all nodes
    # X = empty
    # R = empty
    all_nodes_mask = (1 << N) - 1
    bk(0, all_nodes_mask, 0, 0.0)
    
    # Convert best_clique_mask to list of values
    result_indices = mask_to_list(best_clique_mask)
    result = [values[i] for i in result_indices]
    
    if not result:
        return [2]
        
    # Verify
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if lcm(result[i], result[j]) <= 256:
                 # Fallback
                 greedy_indices = mask_to_list(R_greedy_mask)
                 return [values[i] for i in greedy_indices]
                 
    return result
