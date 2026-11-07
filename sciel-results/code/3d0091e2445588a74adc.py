import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute validity of all subsets: is it a clique or independent set?
    # valid[mask] = 1 if clique, 2 if independent, 0 if neither
    valid = [0] * (1 << n)
    valid[0] = 1  # empty set is both
    
    # For each vertex, precompute which other vertices it's adjacent to
    # A set is a clique iff for every v in set, all other vertices in set are in adj[v]
    # A set is independent iff for every v in set, no other vertex in set is in adj[v]
    
    # Efficient check: use the property that if mask is clique, then for the lowest vertex v,
    # (mask ^ (1<<v)) must be subset of adj_mask[v], and (mask ^ (1<<v)) must also be a clique
    # Similarly for independent sets
    
    # Precompute using DP
    # For clique: mask is clique iff lowest bit v: (mask without v) is clique AND (mask without v) subset of adj_mask[v]
    # For independent: mask is independent iff lowest bit v: (mask without v) is independent AND (mask without v) disjoint from adj_mask[v]
    
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj_mask[v]) == 0:
            is_clique[mask] = True
        if is_indep[rest] and (rest & adj_mask[v]) == 0:
            is_indep[mask] = True
        
        if is_clique[mask] or is_indep[mask]:
            valid[mask] = 1
    
    # Greedy upper bound: assign each vertex to first valid part
    def greedy_upper_bound():
        parts = []  # list of bitmasks
        for v in range(n):
            vmask = 1 << v
            placed = False
            for i, part in enumerate(parts):
                trial = part | vmask
                if valid[trial]:
                    parts[i] = trial
                    placed = True
                    break
            if not placed:
                parts.append(vmask)
        return len(parts)
    
    upper = greedy_upper_bound()
    
    # Lower bound: at least 1, and we can use chromatic number of complement or something
    # Simple lower bound: 1
    # Better: if no valid partition with k parts exists, we need more
    # Let's just try from 1 upward
    
    deadline = time.time() + 85.0
    
    # Backtracking: assign vertices one by one
    # State: which part each vertex is in (0..k-1), or -1 if unassigned
    # We use symmetry breaking: when assigning vertex v, only try:
    #   - existing parts 0..m-1 (where m is number of used parts)
    #   - part m (first new part)
    
    # Order vertices by degree (descending) for better pruning
    degrees = [bin(adj_mask[v]).count('1') for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    # For each k, try to find a valid partition
    # Use memoization: failed_states = set of (assigned_mask, k) that failed
    # But this might be too large. Instead, just use deadline.
    
    best_k = upper
    best_partition = None
    
    # Try k from 1 to upper
    for k in range(1, upper + 1):
        if time.time() > deadline:
            break
        
        # parts[i] = bitmask of vertices in part i
        parts = [0] * k
        # assigned[v] = part index or -1
        assigned = [-1] * n
        used_parts = 0  # number of non-empty parts
        
        # failed: set of (assigned_mask, used_parts) that led to dead end
        # But this could be huge. Let's just rely on deadline.
        
        found = False
        
        def backtrack(idx):
            nonlocal found, used_parts
            if time.time() > deadline:
                return False
            
            if idx == n:
                found = True
                return True
            
            v = vertex_order[idx]
            vmask = 1 << v
            
            # Try existing parts
            for i in range(used_parts):
                trial = parts[i] | vmask
                if valid[trial]:
                    parts[i] = trial
                    assigned[v] = i
                    if backtrack(idx + 1):
                        return True
                    parts[i] = trial ^ vmask
                    assigned[v] = -1
            
            # Try first new part
            if used_parts < k:
                parts[used_parts] = vmask
                assigned[v] = used_parts
                used_parts += 1
                if backtrack(idx + 1):
                    return True
                parts[used_parts - 1] = 0
                assigned[v] = -1
                used_parts -= 1
            
            return False
        
        if backtrack(0):
            best_k = k
            # Extract partition
            part_masks = [parts[i] for i in range(used_parts) if parts[i] != 0]
            best_partition = part_masks
            break
    
    # If no partition found within deadline, use greedy
    if best_partition is None:
        # Reconstruct greedy
        parts = []
        for v in range(n):
            vmask = 1 << v
            placed = False
            for i, part in enumerate(parts):
                trial = part | vmask
                if valid[trial]:
                    parts[i] = trial
                    placed = True
                    break
            if not placed:
                parts.append(vmask)
        best_partition = parts
    
    # Convert to list of lists of ints
    final_result = []
    for mask in best_partition:
        verts = []
        temp = mask
        while temp:
            lsb = temp & (-temp)
            verts.append(lsb.bit_length() - 1)
            temp ^= lsb
        verts.sort()
        final_result.append(verts)
    
    return final_result

# EVOLVE-BLOCK-END
