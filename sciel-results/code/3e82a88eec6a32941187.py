"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute is_clique and is_ind
    is_clique = [False] * size
    is_ind = [False] * size
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        is_clique[mask] = is_clique[rest] and (rest & ~adj[v]) == 0
        is_ind[mask] = is_ind[rest] and (rest & (full_mask ^ adj[v])) == 0
    
    # Collect all valid non-empty subsets (cliques or independent sets)
    # Group by their lowest set bit for efficient DP
    valid_subsets_by_lb = [[] for _ in range(n)]
    for mask in range(1, size):
        if is_clique[mask] or is_ind[mask]:
            lb_bit = (mask & (-mask)).bit_length() - 1
            valid_subsets_by_lb[lb_bit].append(mask)
    
    # DP: dp[mask] = min parts to cover mask
    # For each mask, the lowest set bit must be covered by some valid subset containing it
    dp = [255] * size
    dp[0] = 0
    choice = [0] * size
    
    for mask in range(1, size):
        if time.time() - t0 > DEADLINE:
            break
        lb_bit = (mask & (-mask)).bit_length() - 1
        best = 255
        best_sub = 0
        
        # Only check valid subsets that contain lb_bit and are subsets of mask
        for candidate in valid_subsets_by_lb[lb_bit]:
            if (candidate & mask) == candidate:  # candidate is subset of mask
                prev = mask ^ candidate
                val = dp[prev] + 1
                if val < best:
                    best = val
                    best_sub = candidate
                    if best == 1:
                        break
        
        dp[mask] = best
        choice[mask] = best_sub
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        if sub == 0:
            sub = mask & (-mask)
        vertices = [i for i in range(n) if sub & (1 << i)]
        parts.append(vertices)
        mask ^= sub
    
    result = [[int(v) for v in part] for part in parts]
    return result
# EVOLVE-BLOCK-END
