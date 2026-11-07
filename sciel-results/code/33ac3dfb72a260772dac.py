import time

def _can_match_optimized(m, L):
    """
    Check if (m, m+L] contains distinct b_1...b_12 with i | b_i.
    Uses optimized bipartite matching (Kuhn's algorithm) with heuristic ordering.
    """
    if L < 12:
        return False
        
    # Pre-compute candidate counts for each index i (1..12)
    # to sort indices by constraint tightness (most constrained first)
    indices_info = []
    for i in range(1, 13):
        count = 0
        # v ranges from m+1 to m+L
        # v % i == 0
        # First multiple of i >= m+1
        # k = ceil((m+1)/i) * i
        # We just count directly for simplicity, L is small
        for v in range(m + 1, m + L + 1):
            if v % i == 0:
                count += 1
        indices_info.append((count, i))
        
    # Sort by count ascending
    indices_info.sort()
    
    # Early exit if any index has 0 candidates
    if indices_info[0][0] == 0:
        return False
        
    # match_idx[k] = i if the k-th value (m+1+k) is assigned to index i
    # k ranges from 0 to L-1
    match_idx = [-1] * L
    
    # Recursive DFS to find augmenting path for index i
    # visited[k] tracks if value index k has been visited in current DFS
    def dfs(i, visited):
        for k in range(L):
            v = m + 1 + k
            if v % i != 0:
                continue
            if visited[k]:
                continue
            visited[k] = True
            # If value k is unassigned, or the index currently assigned to it can be rematched
            if match_idx[k] == -1 or dfs(match_idx[k], visited):
                match_idx[k] = i
                return True
        return False

    for count, i in indices_info:
        visited = [False] * L
        if not dfs(i, visited):
            return False
            
    return True

def construct():
    """
    Find m in 0..1000000 that maximizes f(12, m).
    f(12, m) is the least positive L such that (m, m+L] contains a valid matching.
    f(12, m) is periodic with period lcm(1..12) = 27720.
    We search m in [0, 27720).
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    K = 27720
    best_m = 0
    best_L = 0
    
    # We can parallelize or just loop. Loop is fine if fast enough.
    for m in range(K):
        if time.time() > deadline:
            break
        
        # Find least L
        # L must be at least 12 (pigeonhole, distinct values)
        # Max L is likely < 200
        for L in range(12, 200):
            if _can_match_optimized(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                break
                
    return int(best_m)
