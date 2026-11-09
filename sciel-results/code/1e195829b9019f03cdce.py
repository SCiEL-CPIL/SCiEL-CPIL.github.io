import time

P = 27720  # LCM(1..12)

def precompute_masks():
    masks = [0] * P
    for r in range(P):
        m = 0
        for i in range(1, 13):
            if r % i == 0:
                m |= (1 << (i - 1))
        masks[r] = m
    return masks

MASKS = precompute_masks()

def has_matching(m, L):
    base = m + 1
    adj = [[] for _ in range(12)]
    
    for j in range(L):
        res = (base + j) % P
        mask = MASKS[res]
        if mask & 1: adj[0].append(j)
        if mask & 2: adj[1].append(j)
        if mask & 4: adj[2].append(j)
        if mask & 8: adj[3].append(j)
        if mask & 16: adj[4].append(j)
        if mask & 32: adj[5].append(j)
        if mask & 64: adj[6].append(j)
        if mask & 128: adj[7].append(j)
        if mask & 256: adj[8].append(j)
        if mask & 512: adj[9].append(j)
        if mask & 1024: adj[10].append(j)
        if mask & 2048: adj[11].append(j)
    
    match_to = [-1] * L
    match_from = [-1] * 12
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_to[j] == -1 or dfs(match_to[j], visited):
                match_to[j] = i
                match_from[i] = j
                return True
        return False
    
    for i in range(12):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    
    return True

def lower_bound_L(m):
    """Compute the minimum L such that each i in 1..12 has at least one multiple in (m, m+L]"""
    max_gap = 0
    for i in range(1, 13):
        r = m % i
        if r == 0:
            gap = i
        else:
            gap = i - r
        if gap > max_gap:
            max_gap = gap
    return max_gap

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_m = 0
    best_L = 0
    
    for m in range(P):
        if time.time() > deadline:
            break
        
        # Quick lower bound check
        lb = lower_bound_L(m)
        if lb <= best_L:
            continue  # Can't beat best_L
        
        # Binary search for min L, starting from lb
        lo, hi = lb, 144
        min_L = 145
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if min_L > best_L:
            best_L = min_L
            best_m = m
            if best_L == 144:
                break
    
    return best_m
