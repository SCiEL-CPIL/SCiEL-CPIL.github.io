"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
    
    # Precompute for each subset: is it a clique? is it an independent set?
    # Use bitmask representation
    full_mask = (1 << n) - 1
    
    # is_clique[mask]: True if vertices in mask form a clique
    # is_ind[mask]: True if vertices in mask form an independent set
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    
    # Single vertices: both clique and independent set
    for v in range(n):
        is_clique[1 << v] = True
        is_ind[1 << v] = True
    
    # Build up: for a set S, S is a clique iff for any v in S, S\{v} is a clique and v is adjacent to all in S\{v}
    # S is an independent set iff for any v in S, S\{v} is an independent set and v is non-adjacent to all in S\{v}
    
    for mask in range(1, 1 << n):
        # Find the lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if rest == 0:
            continue  # already handled
        
        # Check clique: v must be adjacent to all vertices in rest
        if is_clique[rest]:
            # Check v adjacent to all in rest
            clique_ok = True
            m = rest
            while m:
                lsb2 = m & (-m)
                u = lsb2.bit_length() - 1
                if not adj[v][u]:
                    clique_ok = False
                    break
                m ^= lsb2
            if clique_ok:
                is_clique[mask] = True
        
        # Check independent set: v must be non-adjacent to all vertices in rest
        if is_ind[rest]:
            ind_ok = True
            m = rest
            while m:
                lsb2 = m & (-m)
                u = lsb2.bit_length() - 1
                if adj[v][u]:
                    ind_ok = False
                    break
                m ^= lsb2
            if ind_ok:
                is_ind[mask] = True
    
    # DP: dp[mask] = minimum number of parts to cover vertices in mask
    # dp[0] = 0
    # dp[mask] = min over all valid subsets S of mask: dp[mask ^ S] + 1
    # where S is a clique or independent set
    
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [0] * (1 << n)  # parent[mask] = the subset S chosen to cover part of mask
    
    # Iterate over all masks
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        # Try all subsets S of mask that are clique or independent set
        # We can iterate over submasks of mask
        sub = mask
        best = INF
        best_sub = 0
        while sub > 0:
            if is_clique[sub] or is_ind[sub]:
                val = dp[mask ^ sub] + 1
                if val < best:
                    best = val
                    best_sub = sub
            sub = (sub - 1) & mask
        
        dp[mask] = best
        parent[mask] = best_sub
    
    # Reconstruct the partition
    partition = []
    mask = full_mask
    while mask > 0:
        if time.time() > deadline:
            break
        sub = parent[mask]
        if sub == 0:
            # Fallback: should not happen, but add remaining as singletons
            while mask > 0:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                partition.append([v])
                mask ^= lsb
            break
        # Extract vertices from sub
        verts = []
        m = sub
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        partition.append(verts)
        mask ^= sub
    
    # Ensure all vertices are covered exactly once
    covered = [0] * n
    for part in partition:
        for v in part:
            covered[v] += 1
    
    # If something went wrong, fall back to greedy
    if any(c != 1 for c in covered):
        # Greedy fallback
        es = {tuple(e) for e in P['edges']}
        es |= {(e[1], e[0]) for e in P['edges']}
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    return partition

# EVOLVE-BLOCK-END
