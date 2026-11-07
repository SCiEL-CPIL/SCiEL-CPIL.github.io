import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
        
    # Precompute is_clique and is_ind for all masks
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_hom = [False] * (1 << n)
    
    is_clique[0] = True
    is_ind[0] = True
    is_hom[0] = True # Empty set is both, but we won't use it as a part
    
    # Iterate through masks
    # To speed up, we can compute based on subsets
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest]:
            # Check if v is connected to all in rest
            is_c = True
            m = rest
            while m:
                u_lsb = m & -m
                u = u_lsb.bit_length() - 1
                if not adj[v][u]:
                    is_c = False
                    break
                m ^= u_lsb
            if is_c:
                is_clique[mask] = True
        
        # Check independent set
        if is_ind[rest]:
            # Check if v is NOT connected to any in rest
            is_i = True
            m = rest
            while m:
                u_lsb = m & -m
                u = u_lsb.bit_length() - 1
                if adj[v][u]:
                    is_i = False
                    break
                m ^= u_lsb
            if is_i:
                is_ind[mask] = True
                
        if is_clique[mask] or is_ind[mask]:
            is_hom[mask] = True

    memo = {}
    choice = {}
    
    def can_solve(mask, k):
        if mask == 0:
            return True
        if k == 0:
            return False
        if (mask, k) in memo:
            return memo[(mask, k)]
        
        # Lower bound: if number of vertices > k * max_possible_part_size? 
        # Max part size is n, so not helpful.
        # Simple bound: if k * n < popcount(mask): return False (always true since k>=1)
        
        # Pick lowest set bit vertex
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Iterate subsets R of rest
        # S = lsb | R
        sub = rest
        # We want to iterate all subsets of rest.
        # Standard trick: sub = (sub-1) & rest
        # Start with sub = rest, then sub = (sub-1) & rest, etc.
        # But we need to include R=0 (S=lsb) as well.
        # The loop `while True` handles this if we start correctly or handle 0 case.
        
        # To ensure we try smaller sets first (greedy-like heuristic might help prune), 
        # or larger sets? 
        # Usually, covering more vertices with one part reduces k faster.
        # Let's try larger subsets first? 
        # Iterating subsets in decreasing order of size is complex.
        # Let's just iterate all.
        
        found = False
        # Iterate all subsets R of rest
        R = rest
        while True:
            S = lsb | R
            if is_hom[S]:
                if can_solve(mask ^ S, k - 1):
                    memo[(mask, k)] = True
                    choice[(mask, k)] = S
                    return True
            if R == 0:
                break
            R = (R - 1) & rest
            
        memo[(mask, k)] = False
        return False

    # Find minimum k
    min_k = -1
    for k in range(1, n + 1):
        if can_solve(full_mask, k):
            min_k = k
            break
            
    if min_k == -1:
        # Fallback to greedy if something went wrong (should not happen)
        # But we expect to find it.
        return [[i] for i in range(n)]

    # Reconstruct partition
    parts = []
    mask = full_mask
    k = min_k
    while mask != 0:
        S = choice[(mask, k)]
        # Extract vertices from S
        verts = []
        m = S
        while m:
            lsb_m = m & -m
            v_m = lsb_m.bit_length() - 1
            verts.append(v_m)
            m ^= lsb_m
        parts.append(verts)
        mask ^= S
        k -= 1
        
    return parts
