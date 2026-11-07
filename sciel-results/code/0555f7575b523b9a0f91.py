import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute clique and independent set masks
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
        
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
    
    valid = [False] * (1 << n)
    for mask in range(1 << n):
        valid[mask] = is_clique[mask] or is_ind[mask]
    
    # DP for cochromatic number
    INF = 100
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)
    
    deadline = time.time() + 85.0
    completed = True
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            completed = False
            break
        best = INF
        best_sub = -1
        sub = mask
        while sub > 0:
            if valid[sub]:
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best:
                    best = val
                    best_sub = sub
            sub = (sub - 1) & mask
        dp[mask] = best
        parent[mask] = best_sub
    
    if completed and dp[full_mask] < INF:
        partition = []
        mask = full_mask
        while mask > 0:
            sub = parent[mask]
            part = [int(i) for i in range(n) if (sub >> i) & 1]
            partition.append(part)
            mask ^= sub
        return partition
    
    # Fallback: greedy if DP didn't complete
    # This shouldn't happen for n=14, but just in case
    remaining = full_mask
    partition = []
    while remaining:
        # Find largest valid subset of remaining
        best_sub = 0
        best_size = 0
        sub = remaining
        while sub > 0:
            if valid[sub]:
                sz = sub.bit_count()
                if sz > best_size:
                    best_size = sz
                    best_sub = sub
            sub = (sub - 1) & remaining
        if best_sub == 0:
            # Single vertex is always valid (clique of size 1)
            lsb = remaining & (-remaining)
            best_sub = lsb
        part = [int(i) for i in range(n) if (best_sub >> i) & 1]
        partition.append(part)
        remaining ^= best_sub
    
    return partition
# EVOLVE-BLOCK-END
