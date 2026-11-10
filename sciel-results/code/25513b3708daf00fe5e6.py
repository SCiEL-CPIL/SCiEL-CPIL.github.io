import time

def construct():
    start_time = time.time()
    # 55s limit to be safe
    deadline = start_time + 55.0
    
    N = P['n']
    full_mask = (1 << N) - 1
    
    adj = [0] * N
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    size = 1 << N
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        v_bit = mask & -mask
        v = v_bit.bit_length() - 1
        rest = mask ^ v_bit
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        if is_indep[rest] and (rest & adj[v]) == 0:
            is_indep[mask] = True
            
    dp = [float('inf')] * size
    choice = [0] * size
    dp[0] = 0
    
    timed_out = False
    for mask in range(1, size):
        if time.time() > deadline:
            timed_out = True
            break
        best = float('inf')
        best_sub = 0
        sub = mask
        while sub:
            if is_clique[sub] or is_indep[sub]:
                val = 1 + dp[mask ^ sub]
                if val < best:
                    best = val
                    best_sub = sub
            sub = (sub - 1) & mask
        dp[mask] = best
        choice[mask] = best_sub
        
    if timed_out:
        # Fallback to Greedy
        es = {tuple(e) for e in P['edges']}
        parts = []
        for v in range(N):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts

    parts = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        part = []
        temp = sub
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            part.append(v)
            temp ^= v_bit
        parts.append(part)
        mask ^= sub
        
    return parts
