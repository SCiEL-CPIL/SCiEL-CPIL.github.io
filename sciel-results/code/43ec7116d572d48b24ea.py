import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute clique and independent set status for all masks
    # is_clique[mask] is True if mask induces a clique
    # is_indep[mask] is True if mask induces an independent set
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Find the lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        # adj[v] & rest == rest means v is connected to all vertices in rest
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent: v must NOT be connected to any in rest, and rest must be independent
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True
            
    # valid[mask] is True if mask is a valid part (clique or independent)
    valid = [is_clique[m] or is_indep[m] for m in range(1 << n)]
    
    # Helper to get list of vertices from mask
    def mask_to_list(mask):
        return [i for i in range(n) if mask & (1 << i)]
        
    # DFS with memoization to check if mask can be partitioned into k valid parts
    # We also store the choice of the first part to allow reconstruction
    memo = {}
    
    def can_partition(mask, k):
        if mask == 0:
            return k == 0
        if k == 0:
            return mask == 0
        if k > bin(mask).count('1'):
            return False
            
        key = (mask, k)
        if key in memo:
            return memo[key]
            
        # Symmetry breaking: The part containing the lowest set bit of mask
        # must be chosen first. Let v be the lowest set bit.
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        
        # Iterate over all submasks 'sub' of mask that contain v
        # sub must be a valid part
        # We iterate submasks of mask.
        # To iterate submasks containing v:
        # Let rest = mask ^ lsb.
        # sub = lsb | sub_rest, where sub_rest is a submask of rest.
        rest = mask ^ lsb
        
        sub_rest = rest
        found = False
        while True:
            sub = lsb | sub_rest
            if valid[sub]:
                if can_partition(mask ^ sub, k - 1):
                    memo[key] = True
                    # Store the choice for reconstruction
                    # We need a separate structure or return the path
                    # For now, just return True. We will reconstruct in a second pass or store choices.
                    # Let's store the choice in a dict 'choice'
                    choice[key] = sub
                    return True
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest
            
        memo[key] = False
        return False

    # To support reconstruction, we need to store the specific 'sub' chosen.
    # Let's re-implement can_partition to store choices in a global dict.
    choice = {}
    memo = {}
    
    def dfs(mask, k):
        if mask == 0:
            return k == 0
        if k == 0:
            return mask == 0
        if k > bin(mask).count('1'):
            return False
            
        key = (mask, k)
        if key in memo:
            return memo[key]
            
        lsb = mask & -mask
        rest = mask ^ lsb
        
        sub_rest = rest
        while True:
            sub = lsb | sub_rest
            if valid[sub]:
                if dfs(mask ^ sub, k - 1):
                    memo[key] = True
                    choice[key] = sub
                    return True
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest
            
        memo[key] = False
        return False

    # Find minimum k
    best_k = None
    for k in range(1, n + 1):
        if dfs(full_mask, k):
            best_k = k
            break
            
    if best_k is None:
        # Fallback: single vertex parts (always valid, independent sets of size 1)
        # This should never happen as k=n is always valid.
        return [[i] for i in range(n)]
        
    # Reconstruct partition
    partition = []
    mask = full_mask
    k = best_k
    while k > 0:
        key = (mask, k)
        sub = choice[key]
        partition.append(mask_to_list(sub))
        mask ^= sub
        k -= 1
        
    # Ensure types are correct: list of lists of Python ints
    final_result = []
    for part in partition:
        final_result.append([int(v) for v in part])
        
    return final_result
