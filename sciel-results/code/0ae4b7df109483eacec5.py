import time

# EVOLVE-BLOCK-START
P = 27720  # lcm(1..12)

def _precompute_masks():
    """For each offset j in 0..P-1 (representing position m+1+j), 
    compute bitmask of which i (1..12) divide (m+1+j).
    But it depends on m. Instead, for a fixed m, we compute on the fly.
    Actually, let's precompute for each residue r mod i whether it's 0.
    Better: for each position offset j (0-indexed from m+1), the number is m+1+j.
    (m+1+j) % i == 0 iff j % i == (-m-1) % i.
    So for fixed m, the pattern of which i divide position j is periodic in j with period lcm(1..12)=27720.
    But L <= 144, so we just compute directly.
    """
    pass

def _has_matching(m, L):
    """Check if interval (m, m+L] has distinct b_1..b_12 with i|b_i.
    Positions are m+1, m+2, ..., m+L (L positions).
    We need to assign each i in 1..12 to a distinct position j in 1..L such that i | (m+j).
    """
    if L < 12:
        return False
    
    # Build adjacency: for each i, which positions j (1..L) have i | (m+j)
    # Use DFS-based bipartite matching (Kuhn's algorithm)
    # Left: i in 1..12, Right: j in 1..L
    # adj[i] = list of j values
    
    # For matching, we match i to position j
    # match[j] = i that is matched to position j, or -1
    match_pos = [-1] * (L + 1)  # 1-indexed positions
    
    def dfs(i, visited):
        for j in range(1, L + 1):
            if visited[j]:
                continue
            if (m + j) % i != 0:
                continue
            visited[j] = True
            if match_pos[j] == -1 or dfs(match_pos[j], visited):
                match_pos[j] = i
                return True
        return False
    
    # We need to match all 12 indices
    # Use standard Kuhn: for each left node, try to find augmenting path
    # But we need to be careful: we want to match ALL 12 left nodes
    # Standard approach: try to match each i, using DFS
    
    # Optimize: precompute for each i the list of valid j's
    valid = {}
    for i in range(1, 13):
        valid[i] = []
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                valid[i].append(j)
        if not valid[i]:
            return False  # No position for i
    
    # Try matching: process indices in order of fewest options first (heuristic)
    indices = sorted(range(1, 13), key=lambda i: len(valid[i]))
    
    match_pos = [-1] * (L + 1)
    
    def dfs_match(i, visited):
        for j in valid[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_pos[j] == -1 or dfs_match(match_pos[j], visited):
                match_pos[j] = i
                return True
        return False
    
    for i in indices:
        visited = [False] * (L + 1)
        if not dfs_match(i, visited):
            return False
    
    return True

def _compute_f(m):
    """Compute f(12, m): least positive L such that matching exists."""
    # L must be at least 12
    # Upper bound: 144
    # Binary search? No, because the property isn't monotone in a simple way.
    # Actually, if matching exists for L, it exists for L+1 (we can just ignore the new position).
    # So it IS monotone: if f(12,m) <= L, then for any L' >= L, matching also exists.
    # So binary search works!
    
    lo, hi = 12, 144
    if not _has_matching(m, 144):
        # Shouldn't happen, but return a large value
        return 144
    
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_m = 0
    best_f = 0
    
    # Search all m in [0, P) since f is periodic with period P
    for m in range(P):
        if time.time() > deadline:
            break
        
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Return best_m as Python int
    return int(best_m)
# EVOLVE-BLOCK-END
