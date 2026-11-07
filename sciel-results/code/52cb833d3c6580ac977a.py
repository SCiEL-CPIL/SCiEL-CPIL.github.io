import sys

def construct():
    N = 12
    MOD = 27720
    best_m = 0
    best_L = 0
    
    # To speed up, we can precompute for each i and residue r (m % MOD)
    # the list of offsets d in 1..MAX_L such that (m+d) % i == 0.
    # But m changes, so (m+d) % i depends on m.
    # (m+d) % i == 0 <=> d % i == (-m) % i.
    
    max_L_search = 200 # Safe upper bound
    
    for m in range(MOD):
        # For this m, find minimal L
        found_L = None
        for L in range(1, max_L_search + 1):
            if check_matching(m, L, N):
                found_L = L
                break
        
        if found_L is not None:
            if found_L > best_L:
                best_L = found_L
                best_m = m
            # Early termination if we hit theoretical max? 
            # Theoretical max is likely not 200. 
            # If we find a very large L, we can stop if we are confident.
            # But let's just finish the loop to be sure, it's fast.
            
    return int(best_m)

def check_matching(m, L, N):
    # Indices 1..N
    # Numbers m+1 .. m+L
    # Build adjacency: for each index i, list of numbers in (m, m+L] divisible by i.
    adj = [[] for _ in range(N+1)]
    
    for x in range(m+1, m+L+1):
        for i in range(1, N+1):
            if x % i == 0:
                adj[i].append(x)
                
    # Try to find matching
    # Use DFS based bipartite matching
    # match_to_num[num] = index
    # match_to_idx[idx] = num
    
    # We need to map numbers to an array index for fast access
    # Numbers are in range m+1 .. m+L.
    # We can use a dictionary or offset array.
    # Since L is small, dict is fine.
    
    match_to_idx = {} # num -> idx
    match_to_num = {} # idx -> num
    
    def dfs(idx, visited):
        for num in adj[idx]:
            if num in visited:
                continue
            visited.add(num)
            if num not in match_to_idx or dfs(match_to_idx[num], visited):
                match_to_idx[num] = idx
                match_to_num[idx] = num
                return True
        return False
    
    # Sort indices by degree to improve pruning?
    # Standard Kuhn's algorithm complexity is O(V*E).
    # V=12, E is small.
    
    # To be safe and fast, order indices by size of adj list
    indices = list(range(1, N+1))
    indices.sort(key=lambda i: len(adj[i]))
    
    for idx in indices:
        if not dfs(idx, set()):
            return False
            
    return True
