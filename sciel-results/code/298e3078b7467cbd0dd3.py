def can_match_fast(r, L, N):
    # Map values to indices 0..L-1
    # Value v = r + 1 + idx
    # So idx = v - (r + 1)
    
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        k_min = r // i + 1
        k_max = (r + L) // i
        for k in range(k_min, k_max + 1):
            v = k * i
            idx = v - (r + 1)
            if 0 <= idx < L:
                adj[i].append(idx)
                
    for i in range(1, N + 1):
        if not adj[i]:
            return False
            
    match_to_v = [-1] * L  # match_to_v[idx] = i
    
    def dfs(i, visited):
        for idx in adj[i]:
            if visited & (1 << idx):
                continue
            visited |= (1 << idx)
            if match_to_v[idx] == -1 or dfs(match_to_v[idx], visited):
                match_to_v[idx] = i
                return True
        return False
        
    for i in range(1, N + 1):
        visited = 0
        if not dfs(i, visited):
            return False
    return True
