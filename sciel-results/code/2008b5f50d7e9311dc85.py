import time

def construct():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P['n']
    edges = P['edges']
    
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute valid_part[mask]: True if mask is a clique or independent set
    size = 1 << n
    valid_part = [False] * size
    valid_part[0] = True
    
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        is_clique[mask] = is_clique[rest] and ((adj[v] & rest) == rest)
        is_independent[mask] = is_independent[rest] and ((adj[v] & rest) == 0)
        valid_part[mask] = is_clique[mask] or is_independent[mask]
    
    if time.time() > deadline:
        return [[i] for i in range(n)]
    
    # Subset DP: dp[mask] = minimum number of parts to cover mask
    INF = n + 1
    dp = [INF] * size
    dp[0] = 0
    parent = [-1] * size
    
    for mask in range(1, size):
        if (mask & 0xFF) == 0 and time.time() > deadline:
            break
        
        lsb = mask & -mask
        sub = mask
        while sub:
            if sub & lsb:
                if valid_part[sub]:
                    prev = mask ^ sub
                    if dp[prev] + 1 < dp[mask]:
                        dp[mask] = dp[prev] + 1
                        parent[mask] = sub
            sub = (sub - 1) & mask
    
    full_mask = (1 << n) - 1
    best_num_parts = dp[full_mask]
    
    # Reconstruct the partition
    parts_masks = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub == -1:
            break
        parts_masks.append(sub)
        mask ^= sub
    
    if parts_masks:
        best_parts = []
        for pm in parts_masks:
            verts = [v for v in range(n) if pm & (1 << v)]
            best_parts.append(verts)
    else:
        best_parts = [[i] for i in range(n)]
    
    # Verify
    def verify(parts):
        seen = set()
        for part in parts:
            if not part:
                return False
            for v in part:
                if v in seen:
                    return False
                seen.add(v)
            is_cli = True
            is_ind = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    u, w = part[i], part[j]
                    if adj[u] & (1 << w):
                        is_ind = False
                    else:
                        is_cli = False
                    if not is_cli and not is_ind:
                        return False
        return len(seen) == n
    
    if not verify(best_parts):
        # Fallback greedy
        es = {tuple(e) for e in edges} | {tuple(reversed(e)) for e in edges}
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
        best_parts = parts
    
    return best_parts
