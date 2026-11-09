import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s, we aim to finish well before.
    # DP for N=14 is very fast, but we guard against unexpected issues.
    deadline = start_time + 50.0 

    # 1. Build adjacency bitmasks
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # 2. Precompute valid pieces (clique or independent set)
    # is_valid[mask] is True if the induced subgraph on mask is a clique or an independent set
    is_valid = [False] * (1 << n)
    is_valid[0] = True # Empty set is trivially valid, though we don't use it as a part
    
    # For each mask, check if it's a clique or independent set
    # Optimization: We can check incrementally or just iterate bits.
    # Given N=14, iterating bits for each of 16384 masks is cheap.
    for mask in range(1, 1 << n):
        is_clique = True
        is_indep = True
        
        # Iterate through vertices in the mask
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            rest = mask ^ lsb
            
            # Check clique condition for u: all others in rest must be neighbors of u
            # If there is any vertex in rest that is NOT a neighbor of u, it's not a clique.
            # ~adj[u] has bits set for non-neighbors (and self, but u is not in rest).
            # We need to mask ~adj[u] to only consider vertices in 'rest'.
            if rest & ~adj[u]:
                is_clique = False
            
            # Check indep condition for u: no others in rest must be neighbors of u
            # If there is any vertex in rest that IS a neighbor of u, it's not independent.
            if rest & adj[u]:
                is_indep = False
                
            if not is_clique and not is_indep:
                break
                
            m ^= lsb
            
        is_valid[mask] = is_clique or is_indep

    # 3. DP for minimum number of parts
    # dp[mask] = min parts to partition 'mask'
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    # best_sub[mask] stores the submask chosen as the last part in the optimal partition of 'mask'
    best_sub = [0] * (1 << n)
    
    # Iterate masks from 1 to full_mask
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
            
        # Iterate submasks of 'mask'
        # We want to find a valid submask 'sub' such that dp[mask ^ sub] + 1 is minimized.
        sub = mask
        while sub:
            if is_valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    best_sub[mask] = sub
            sub = (sub - 1) & mask

    # Check if we finished the DP
    if time.time() > deadline or dp[full_mask] == INF:
        # Fallback: Greedy construction
        # This is a safety net. For N=14, DP should always finish.
        parts = []
        remaining = full_mask
        while remaining:
            # Find largest valid subset of remaining
            best = 0
            best_size = 0
            sub = remaining
            while sub:
                if is_valid[sub]:
                    sz = sub.bit_count()
                    if sz > best_size:
                        best_size = sz
                        best = sub
                sub = (sub - 1) & remaining
            
            if best == 0:
                # Should not happen, but handle single vertex
                lsb = remaining & -remaining
                best = lsb
            
            # Extract vertices from 'best'
            part_vertices = []
            m = best
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                part_vertices.append(v)
                m ^= lsb
            parts.append(part_vertices)
            remaining ^= best
            
        # Ensure types are int
        return [[int(x) for x in part] for part in parts]

    # 4. Reconstruct partition from DP table
    parts = []
    current_mask = full_mask
    while current_mask:
        sub = best_sub[current_mask]
        # Extract vertices from 'sub'
        part_vertices = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        parts.append(part_vertices)
        current_mask ^= sub
        
    # Ensure types are int
    return [[int(x) for x in part] for part in parts]

# EVOLVE-BLOCK-END
