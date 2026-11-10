"""Optimized cochromatic partition using Bron-Kerbosch for maximal sets + DP over valid subsets."""

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time

def construct():
    _t0 = time.time()
    _deadline = 75.0  # seconds, safety margin before 90s kill

    n = P['n']
    
    # Precompute adjacency bitmask
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Compute complement adjacency (for finding maximal independent sets in original graph,
    # which are maximal cliques in the complement)
    comp_adj_mask = [0] * n
    for i in range(n):
        comp_adj_mask[i] = ((1 << n) - 1) ^ (1 << i) ^ adj_mask[i]

    def bron_kerbosch(adj, R, P, X, results):
        """Find all maximal cliques using Bron-Kerbosch with pivot selection."""
        if not P and not X:
            results.append(R)
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        # We want to pick a vertex with maximum degree in the induced subgraph on P ∪ X
        # Actually, standard pivot: pick u in P ∪ X that maximizes |P ∩ N(u)|
        # Then iterate over P \ N(u)
        
        if P or X:
            # Pick pivot
            union = P | X
            best_u = 0
            best_count = -1
            temp = union
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                # Count neighbors of u in P
                count = bin(adj[u] & P).count('1')
                if count > best_count:
                    best_count = count
                    best_u = u
                temp ^= lsb
            
            # Iterate over vertices in P that are NOT adjacent to best_u
            candidates = P & ~adj[best_u]
            while candidates:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                bron_kerbosch(adj, R | lsb, P & adj[v], X & adj[v], results)
                P ^= lsb
                X |= lsb
        else:
            if R:
                results.append(R)

    # Find all maximal cliques in G
    max_cliques = []
    full_set = (1 << n) - 1
    bron_kerbosch(adj_mask, 0, full_set, 0, max_cliques)
    
    # Find all maximal independent sets in G (which are maximal cliques in complement)
    max_indeps = []
    bron_kerbosch(comp_adj_mask, 0, full_set, 0, max_indeps)
    
    # Precompute is_clique and is_indep for all subsets
    # A subset S is a clique iff it is contained in some maximal clique
    # A subset S is an independent set iff it is contained in some maximal independent set
    
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    is_clique[0] = True
    is_indep[0] = True
    
    for mc in max_cliques:
        # Mark all subsets of mc as cliques
        sub = mc
        while sub:
            is_clique[sub] = True
            sub = (sub - 1) & mc
    
    for mi in max_indeps:
        # Mark all subsets of mi as independent sets
        sub = mi
        while sub:
            is_indep[sub] = True
            sub = (sub - 1) & mi
    
    # Precompute valid: subset is a valid part if clique or independent
    valid = [is_clique[m] or is_indep[m] for m in range(1 << n)]
    
    # DP for minimum number of parts
    INF = float('inf')
    dp = [INF] * (1 << n)
    choice = [0] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        if time.time() - _t0 > _deadline:
            # Fallback: greedy construction
            break
        sub = mask
        best_val = dp[mask]
        best_sub = 0
        while sub:
            if valid[sub]:
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        dp[mask] = best_val
        choice[mask] = best_sub
    
    # Check if DP completed
    full_mask = (1 << n) - 1
    if dp[full_mask] == INF:
        # Fallback: each vertex as its own part
        return [[i] for i in range(n)]
    
    # Reconstruct the partition
    parts = []
    current_mask = full_mask
    while current_mask:
        part_mask = choice[current_mask]
        part_vertices = [i for i in range(n) if (part_mask >> i) & 1]
        parts.append(part_vertices)
        current_mask ^= part_mask
        
    # Ensure all elements are Python int
    result = [list(p) for p in parts]
    for part in result:
        for i in range(len(part)):
            part[i] = int(part[i])
            
    return result
