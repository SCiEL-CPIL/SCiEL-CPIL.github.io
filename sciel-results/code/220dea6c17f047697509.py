import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = 50.0 # 50s limit, 10s margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Find maximal cliques
    max_cliques = []
    def expand(R, P_mask, X_mask):
        if time.time() > deadline:
            return
        if P_mask == 0 and X_mask == 0:
            if R:
                max_cliques.append(R)
            return
        
        if P_mask | X_mask:
            lsb = (P_mask | X_mask) & -(P_mask | X_mask)
            u = lsb.bit_length() - 1
            candidates = P_mask & ~adj_masks[u]
        else:
            candidates = P_mask
            
        while candidates:
            lsb_v = candidates & -candidates
            v = lsb_v.bit_length() - 1
            expand(R | lsb_v, P_mask & adj_masks[v], X_mask & adj_masks[v])
            P_mask &= ~lsb_v
            X_mask |= lsb_v
            candidates &= ~lsb_v
            
    expand(0, full_mask, 0)
    
    # Find maximal independent sets
    comp_adj_masks = [0] * n
    for i in range(n):
        comp_adj_masks[i] = full_mask ^ (1 << i) ^ adj_masks[i]
        
    max_independent_sets = []
    def expand_comp(R, P_mask, X_mask):
        if time.time() > deadline:
            return
        if P_mask == 0 and X_mask == 0:
            if R:
                max_independent_sets.append(R)
            return
            
        if P_mask | X_mask:
            lsb = (P_mask | X_mask) & -(P_mask | X_mask)
            u = lsb.bit_length() - 1
            candidates = P_mask & ~comp_adj_masks[u]
        else:
            candidates = P_mask
            
        while candidates:
            lsb_v = candidates & -candidates
            v = lsb_v.bit_length() - 1
            expand_comp(R | lsb_v, P_mask & comp_adj_masks[v], X_mask & comp_adj_masks[v])
            P_mask &= ~lsb_v
            X_mask |= lsb_v
            candidates &= ~lsb_v
            
    expand_comp(0, full_mask, 0)
    
    valid_sets = set(max_cliques)
    valid_sets.update(max_independent_sets)
    valid_sets_list = list(valid_sets)
    
    # DP
    size = 1 << n
    dp = [float('inf')] * size
    parent = [-1] * size
    dp[0] = 0
    
    for mask in range(1, size):
        if time.time() > deadline:
            break
        for s in valid_sets_list:
            if (s & mask) == s:
                prev = mask ^ s
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = s
                    
    # If DP didn't finish or failed, fallback to greedy
    if time.time() > deadline or dp[full_mask] == float('inf'):
        # Greedy fallback
        es = {tuple(e) for e in edges}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts

    # Reconstruct
    parts = []
    mask = full_mask
    while mask:
        s = parent[mask]
        if s == -1:
            break
        part = []
        for i in range(n):
            if s & (1 << i):
                part.append(i)
        parts.append(part)
        mask ^= s
        
    return parts
