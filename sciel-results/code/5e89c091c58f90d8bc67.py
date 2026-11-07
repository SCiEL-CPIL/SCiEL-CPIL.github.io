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
    # A set S is a clique iff for every v in S, S \ {v} ⊆ adj[v]
    # A set S is independent iff for every v in S, adj[v] ∩ (S \ {v}) = ∅
    
    # Efficient precomputation using DP:
    # is_clique[mask]: True if mask induces a clique
    # is_ind[mask]: True if mask induces an independent set
    
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Clique: v must be adjacent to all in rest, and rest must be clique
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
        
        # Independent: v must NOT be adjacent to any in rest, and rest must be independent
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
    
    # valid[mask] = is_clique[mask] or is_ind[mask]
    valid = [False] * (1 << n)
    for mask in range(1 << n):
        valid[mask] = is_clique[mask] or is_ind[mask]
    
    # Collect valid masks for faster iteration
    valid_masks = [m for m in range(1, 1 << n) if valid[m]]
    
    # DP: dp[mask] = min number of parts to partition set 'mask'
    INF = 100
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)
    
    deadline = time.time() + 80.0
    completed = True
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            completed = False
            break
        
        best = INF
        best_sub = -1
        # Iterate over valid submasks of mask
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
        # Reconstruct
        partition = []
        mask = full_mask
        while mask > 0:
            sub = parent[mask]
            part = [int(i) for i in range(n) if (sub >> i) & 1]
            partition.append(part)
            mask ^= sub
        return partition
    
    # Fallback: branch and bound
    # Try to find a partition with k parts for k = 1, 2, ...
    # For a given k, check if we can cover all vertices with k valid sets
    
    def can_partition(k):
        """Check if full_mask can be partitioned into k valid sets."""
        # Use backtracking
        remaining = full_mask
        parts = []
        
        def backtrack(rem, depth):
            if depth == k:
                return rem == 0
            if rem == 0:
                return True
            
            # Get a vertex from rem to break symmetry
            # Pick the lowest set bit
            v_bit = rem & (-rem)
            v = v_bit.bit_length() - 1
            
            # Try all valid sets containing v that are subsets of rem
            # Enumerate submasks of rem that contain v
            rest_rem = rem ^ v_bit
            sub = rest_rem
            while True:
                candidate = sub | v_bit
                if valid[candidate]:
                    if backtrack(rem ^ candidate, depth + 1):
                        return True
                if sub == 0:
                    break
                sub = (sub - 1) & rest_rem
            
            return False
        
        return backtrack(remaining, 0)
    
    # Try increasing k
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        if can_partition(k):
            # Now reconstruct the actual partition
            # Redo with recording
            def reconstruct_partition(k_parts):
                remaining = full_mask
                parts = []
                
                def backtrack(rem, depth):
                    if depth == k_parts:
                        return rem == 0
                    if rem == 0:
                        return True
                    
                    v_bit = rem & (-rem)
                    v = v_bit.bit_length() - 1
                    rest_rem = rem ^ v_bit
                    sub = rest_rem
                    while True:
                        candidate = sub | v_bit
                        if valid[candidate]:
                            parts.append(candidate)
                            if backtrack(rem ^ candidate, depth + 1):
                                return True
                            parts.pop()
                        if sub == 0:
                            break
                        sub = (sub - 1) & rest_rem
                    return False
                
                if backtrack(remaining, 0):
                    result = []
                    for pm in parts:
                        part = [int(i) for i in range(n) if (pm >> i) & 1]
                        result.append(part)
                    return result
                return None
            
            result = reconstruct_partition(k)
            if result is not None:
                return result
    
    # Final fallback: greedy
    parts = []
    used = [False] * n
    verts = sorted(range(n), key=lambda x: adj[x].bit_count(), reverse=True)
    
    for v in verts:
        if used[v]:
            continue
        placed = False
        for i, part in enumerate(parts):
            trial = part + [v]
            is_c = True
            is_i = True
            for idx in range(len(trial)):
                u = trial[idx]
                for jdx in range(idx + 1, len(trial)):
                    w = trial[jdx]
                    if adj[u] & (1 << w):
                        is_i = False
                    else:
                        is_c = False
            if is_c or is_i:
                parts[i] = trial
                used[v] = True
                placed = True
                break
        if not placed:
            parts.append([v])
            used[v] = True
    
    return parts

# EVOLVE-BLOCK-END
