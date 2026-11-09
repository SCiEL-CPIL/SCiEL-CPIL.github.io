"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        
    size = 1 << n
    full_mask = size - 1
    
    # Precompute valid_mask: True if subset is clique or independent set
    is_clique = [False] * size
    is_ind = [False] * size
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        # Check clique
        # rest must be clique and all vertices in rest must be neighbors of v
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
            
        # Check independent set
        # rest must be independent and all vertices in rest must be non-neighbors of v
        # ~adj[v] within n bits is (full_mask ^ adj[v])
        if is_ind[rest] and (rest & (full_mask ^ adj[v])) == 0:
            is_ind[mask] = True
            
        if is_clique[mask] or is_ind[mask]:
            pass # valid_mask logic handled in DP or precomputed

    # DP: dp[mask] = min number of parts to cover vertices in 'mask'
    dp = [100] * size
    dp[0] = 0
    choice = [0] * size
    
    for mask in range(1, size):
        lb = mask & (-mask)
        rest_mask = mask ^ lb
        best = 100
        best_sub = 0
        
        sub = rest_mask
        while True:
            candidate = sub | lb
            if is_clique[candidate] or is_ind[candidate]:
                prev = mask ^ candidate
                val = dp[prev] + 1
                if val < best:
                    best = val
                    best_sub = candidate
            if sub == 0:
                break
            sub = (sub - 1) & rest_mask
            
        dp[mask] = best
        choice[mask] = best_sub
        
    min_parts = dp[full_mask]
    
    # Reconstruct
    parts = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        if sub == 0:
            sub = mask & (-mask)
        vertices = [i for i in range(n) if sub & (1 << i)]
        parts.append(vertices)
        mask ^= sub
        
    result = [list(part) for part in parts]
    return result
# EVOLVE-BLOCK-END
