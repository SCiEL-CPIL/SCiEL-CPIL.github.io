import time

def construct():
    # Data from the seed program
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind for all subsets
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
            
    is_valid = [is_clique[m] or is_ind[m] for m in range(1 << n)]
    
    # Helper to convert mask to list of vertices
    def mask_to_list(mask):
        verts = []
        while mask:
            b = mask & -mask
            verts.append(b.bit_length() - 1)
            mask ^= b
        return verts

    # Memoization for the search
    # State: (remaining_mask, parts_count) -> boolean (can we finish with <= best_parts_count?)
    # Actually, we want to find the *minimum* number.
    # Let's use a DFS that returns the minimum number of parts needed for 'rem'.
    # But since we need the actual partition, we might just do a standard branch and bound.
    # However, memoization on (rem, current_count) is tricky because current_count varies.
    # Instead, we can compute min_parts[rem] = minimum number of valid subsets to cover 'rem'.
    # This is a standard DP: min_parts[rem] = 1 + min(min_parts[rem ^ sub] for sub in valid_subsets(rem))
    # We can compute this bottom-up or top-down with memoization.
    # Since n=14, 2^14 states is small.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Top-down DP for minimum number of parts
    memo = {}
    
    def min_parts(rem):
        if rem == 0:
            return 0
        if rem in memo:
            return memo[rem]
        
        # Lower bound: at least 1
        # Upper bound: number of bits in rem
        # We want to find the minimum k such that rem can be covered by k valid sets.
        # Iterate over all valid submasks 'sub' of 'rem' that contain the lowest set bit of 'rem'.
        # Why lowest set bit? To avoid symmetry/permutations of the same set of parts.
        # If we always pick a set containing the lowest remaining vertex, we don't double count partitions.
        
        lsb = rem & -rem
        v = lsb.bit_length() - 1
        
        best = n # max possible
        
        # Iterate submasks of rem
        sub = rem
        while sub:
            if (sub & lsb) and is_valid[sub]:
                # If sub is valid and contains v
                # The rest is rem ^ sub
                # Cost = 1 + min_parts(rem ^ sub)
                rest = rem ^ sub
                val = 1 + min_parts(rest)
                if val < best:
                    best = val
            sub = (sub - 1) & rem
            
        memo[rem] = best
        return best

    # Calculate the optimal number of parts
    optimal_k = min_parts(full_mask)
    
    # Now we need to reconstruct the partition.
    # We can do a second pass, guided by the DP values, to pick the actual sets.
    # We want to find a sequence of valid sets S1, S2, ..., Sk such that:
    # 1. Union is full_mask
    # 2. Disjoint
    # 3. Each Si is valid
    # 4. k = optimal_k
    
    # We can do a DFS again, but this time we know the target k.
    # Or we can just follow the choices that led to the minimum in the DP.
    # Let's do a simple recursive reconstruction.
    
    current_parts = []
    
    def reconstruct(rem, k_remaining):
        if rem == 0:
            return True
        if k_remaining == 0:
            return False
            
        lsb = rem & -rem
        v = lsb.bit_length() - 1
        
        # Try all valid submasks containing v
        sub = rem
        while sub:
            if (sub & lsb) and is_valid[sub]:
                rest = rem ^ sub
                # Check if the rest can be covered by k_remaining - 1 parts
                # We need to be careful: min_parts(rest) <= k_remaining - 1
                # But we also need to ensure that the specific path we take leads to a solution.
                # Since min_parts is the *minimum*, if min_parts(rest) <= k_remaining - 1, 
                # there exists a solution.
                if min_parts(rest) <= k_remaining - 1:
                    # Pick this sub
                    current_parts.append(mask_to_list(sub))
                    if reconstruct(rest, k_remaining - 1):
                        return True
                    current_parts.pop()
            sub = (sub - 1) & rem
            
        return False

    # Execute reconstruction
    # We don't strictly need a deadline guard here if the DP is fast enough, 
    # but let's add one just in case, though 2^14 is tiny.
    start_time = time.time()
    deadline = start_time + 5.0
    
    if time.time() < deadline:
        reconstruct(full_mask, optimal_k)
    else:
        # Fallback if something went wrong (should not happen)
        current_parts = None

    if current_parts is None or len(current_parts) != optimal_k:
        # Fallback to the previous greedy or a simple valid partition
        # Greedy: just put each vertex in a new part? No, that's n parts.
        # Let's use the greedy from the seed as a last resort valid partition.
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        current_parts = parts

    # Ensure types are correct
    # The lists contain Python ints from bit_length
    return current_parts
