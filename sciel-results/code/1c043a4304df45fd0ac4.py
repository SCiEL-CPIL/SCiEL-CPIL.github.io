import time

def _lcm(a, b):
    g = 1
    # Euclidean gcd
    while b:
        a, b = b, a % b
    return a * b // g

def _compute_lcm_range(n):
    result = 1
    for i in range(1, n + 1):
        result = _lcm(result, i)
    return result

def _has_perfect_matching(adj, L, n):
    """adj[i] is bitmask of valid positions (1-indexed, bit j-1 for position j) for i in 1..n.
    Check if perfect matching exists for left side 1..n into right side 1..L."""
    if L < n:
        return False
    # DFS-based bipartite matching (Kuhn's algorithm)
    # match_to[i] = which position is matched to index i (0 if unmatched)
    match_to = [0] * (n + 1)
    # For each index, try to find augmenting path
    
    def dfs(i, visited):
        for j in range(1, L + 1):
            if not (adj[i] & (1 << (j - 1))):
                continue
            if visited[j]:
                continue
            visited[j] = True
            if match_to[i] == 0 or dfs(match_to[i], visited):
                match_to[i] = j
                return True
        return False
    
    # Wait, this DFS formulation is wrong. Let me use standard Kuhn's algorithm.
    # match_pos[j] = which index is matched to position j
    # For each index i, try to assign it to some position
    
    match_pos = [0] * (L + 1)  # match_pos[j] = index matched to position j
    
    def try_assign(i, visited):
        for j in range(1, L + 1):
            if not (adj[i] & (1 << (j - 1))):
                continue
            if visited[j]:
                continue
            visited[j] = True
            if match_pos[j] == 0 or try_assign(match_pos[j], visited):
                match_pos[j] = i
                return True
        return False
    
    matched = 0
    for i in range(1, n + 1):
        visited = [False] * (L + 1)
        if try_assign(i, visited):
            matched += 1
        else:
            return False
    return matched == n

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 12
    period = _compute_lcm_range(n)  # 27720
    L_max = 144
    
    best_m = 12
    best_L = 0
    
    # For each residue r in 0..period-1, compute the minimum L
    # Precompute adjacency bitmasks for each residue and each index
    # adj[r][i] = bitmask of positions j in 1..L_max where (r+j) % i == 0
    
    # To save memory and time, process residues in batches
    for r in range(period):
        if time.time() > deadline:
            break
        
        # Build adjacency for this residue
        adj = [0] * (n + 1)
        valid = True
        for i in range(1, n + 1):
            mask = 0
            for j in range(1, L_max + 1):
                if (r + j) % i == 0:
                    mask |= (1 << (j - 1))
            adj[i] = mask
            if mask == 0:
                valid = False
                break
        
        if not valid:
            continue
        
        # Binary search for minimum L
        lo, hi = 1, L_max
        min_L = L_max + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            # Check if perfect matching exists with L=mid
            # Need to mask adj to only positions 1..mid
            if _has_perfect_matching(adj, mid, n):
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if min_L <= L_max and min_L > best_L:
            best_L = min_L
            best_m = r  # r is in 0..period-1, which is in 0..1000000
    
    return best_m
