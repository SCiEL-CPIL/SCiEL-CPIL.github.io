"""Cochromatic partition via subset DP with efficient precomputation."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    n = P['n']
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute is_indep and is_clique for all masks using lowest-set-bit recurrence
    is_indep = [False] * size
    is_clique = [False] * size
    is_indep[0] = True
    is_clique[0] = True
    
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        is_indep[mask] = is_indep[rest] and (adj[v] & rest) == 0
        is_clique[mask] = is_clique[rest] and (adj[v] & rest) == rest
    
    # A mask is a valid part if it's a clique or independent set (and nonempty)
    # is_valid[mask] = True if mask can be a single part
    is_valid = [False] * size
    for mask in range(1, size):
        is_valid[mask] = is_indep[mask] or is_clique[mask]
    
    # DP: dp[mask] = min number of parts to partition vertices in mask
    # parent[mask] = one valid submask chosen as a part in optimal solution
    INF = n + 1
    dp = [INF] * size
    parent = [0] * size
    dp[0] = 0
    
    for mask in range(1, size):
        # Early check: if mask itself is a valid part
        if is_valid[mask]:
            dp[mask] = 1
            parent[mask] = mask
            continue
        
        # Iterate over proper submasks of mask
        # Optimization: iterate submasks in decreasing order of size to find good solutions early
        sub = (mask - 1) & mask
        best_val = INF
        best_sub = 0
        while sub:
            if is_valid[sub]:
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    # Can't do better than 2 if mask isn't valid
                    if best_val == 2:
                        break
            sub = (sub - 1) & mask
        
        dp[mask] = best_val
        parent[mask] = best_sub
        
        # Deadline check
        if time.time() - start_time > deadline:
            # Return best effort so far - reconstruct what we have
            # For masks not yet processed, use fallback
            parts = []
            current = full_mask
            while current:
                if current <= mask and parent[current]:
                    part_mask = parent[current]
                    verts = [i for i in range(n) if (part_mask >> i) & 1]
                    parts.append(verts)
                    current ^= part_mask
                else:
                    # Fallback: single vertex
                    lb = current & (-current)
                    v = lb.bit_length() - 1
                    parts.append([v])
                    current ^= lb
            return parts
    
    # Reconstruct the optimal partition
    parts = []
    current = full_mask
    while current:
        part_mask = parent[current]
        verts = [i for i in range(n) if (part_mask >> i) & 1]
        parts.append(verts)
        current ^= part_mask
    
    # Ensure types: list of lists of Python ints
    return [[int(v) for v in part] for part in parts]
