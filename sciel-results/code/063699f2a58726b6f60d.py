"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj_bitmasks():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _precompute_valid_masks(adj):
    n = P['n']
    size = 1 << n
    is_valid = [False] * size
    is_valid[0] = True # Empty set is technically valid, but we don't use it as a part
    
    # Helper to check if a mask is a clique
    def is_clique(mask):
        if mask == 0:
            return True
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        if (adj[v] & rest) != rest:
            return False
        return is_clique(rest)

    # Helper to check if a mask is an independent set
    def is_independent(mask):
        if mask == 0:
            return True
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        if (adj[v] & rest) != 0:
            return False
        return is_independent(rest)

    # Precompute for all masks
    # Optimization: use iterative DP or simple recursion with memoization? 
    # Recursion depth is n=14, so it's fine. But calling functions 16384 times * 2 is fine.
    # Even better: compute iteratively.
    # Actually, let's just use the recursive helpers, they are fast enough for n=14.
    
    for mask in range(1, size):
        if is_clique(mask) or is_independent(mask):
            is_valid[mask] = True
            
    return is_valid

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    n = P['n']
    adj = _build_adj_bitmasks()
    is_valid = _precompute_valid_masks(adj)
    
    if time.time() > deadline:
        # Fallback to greedy if precomputation took too long (unlikely for n=14)
        return _greedy_fallback(adj)

    size = 1 << n
    full_mask = size - 1
    
    INF = float('inf')
    dp = [INF] * size
    parent = [0] * size # Stores the sub-mask chosen for this mask
    
    dp[0] = 0
    
    # DP over subsets
    # dp[mask] = min parts to cover vertices in 'mask'
    # Transition: dp[mask] = 1 + min(dp[mask ^ sub]) for all valid 'sub' subset of 'mask'
    for mask in range(1, size):
        if time.time() > deadline:
            break
        # Iterate all submasks of mask
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
        
    if dp[full_mask] == INF:
        # Should not happen as singletons are always valid
        return _greedy_fallback(adj)
        
    # Reconstruct partition
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub == 0:
            break # Safety break
        # Convert sub mask to list of vertices
        vs = []
        s = sub
        while s:
            v = (s & -s).bit_length() - 1
            vs.append(v)
            s ^= (1 << v)
        parts.append(vs)
        mask ^= sub
        
    # Convert to proper Python types
    final_parts = []
    for part in parts:
        final_parts.append([int(v) for v in part])
        
    # Ensure all vertices are covered exactly once
    all_vertices = sorted(v for part in final_parts for v in part)
    assert all_vertices == list(range(n)), f"Invalid partition: {all_vertices}"
    
    return final_parts

def _greedy_fallback(adj):
    n = P['n']
    parts = []
    for v in range(n):
        placed = False
        for i, part in enumerate(parts):
            # Check if adding v keeps part as clique or independent
            # This is a bit slow, but it's a fallback
            new_part = part + [v]
            # Check clique
            is_c = True
            for u in new_part:
                if (adj[u] & (1 << v)) == 0:
                    is_c = False
                    break
            if is_c:
                parts[i] = new_part
                placed = True
                break
            # Check independent
            is_i = True
            for u in new_part:
                if (adj[u] & (1 << v)) != 0:
                    is_i = False
                    break
            if is_i:
                parts[i] = new_part
                placed = True
                break
        if not placed:
            parts.append([v])
    return [list(p) for p in parts]

# EVOLVE-BLOCK-END
