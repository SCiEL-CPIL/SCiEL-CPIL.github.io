"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_matching(m, L):
    """
    Check if interval (m, m+L] has distinct b_1..b_12 with i|b_i.
    Uses Kuhn's algorithm for bipartite matching.
    Left side: indices 1..12
    Right side: numbers in (m, m+L]
    """
    n_left = 12
    
    # Identify all candidate numbers in (m, m+L]
    # We only care about numbers divisible by at least one i in 1..12
    # But for matching, we can just use all numbers in the range as potential right nodes,
    # though many will have no edges. It's more efficient to filter.
    
    # Let's build the list of right nodes that are divisible by any i in 1..12
    # Actually, simpler: just use the range m+1 to m+L.
    # Size is L. L <= 144.
    
    right_nodes = list(range(m + 1, m + L + 1))
    n_right = L
    
    # Map value to index in right_nodes
    # Since right_nodes is contiguous, index = val - (m + 1)
    
    # Build adjacency list for left nodes (1..12)
    # adj[i-1] will contain indices of right nodes divisible by i
    adj = [[] for _ in range(n_left)]
    
    for i in range(1, n_left + 1):
        # Find first multiple of i > m
        start = (m // i + 1) * i
        # Iterate through multiples
        k = start
        while k <= m + L:
            # k is in (m, m+L]
            # Index in right_nodes: k - (m + 1)
            idx = k - (m + 1)
            adj[i - 1].append(idx)
            k += i
            
        # If no candidates, matching is impossible
        if not adj[i - 1]:
            return False

    # Kuhn's algorithm
    # match_r[v] = u means right node v is matched to left node u
    match_r = [-1] * n_right
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    for u in range(n_left):
        visited = [False] * n_right
        if not dfs(u, visited):
            return False
            
    return True

def _compute_f(m):
    """
    Compute f(12, m): least L such that (m, m+L] has valid assignment.
    Binary search on L.
    """
    # Lower bound: max distance to next multiple for any i
    lo = 1
    for i in range(1, 13):
        next_mult = (m // i + 1) * i
        dist = next_mult - m
        if dist > lo:
            lo = dist
            
    hi = 144 # Max possible L is 144 per task description
    
    # Ensure lo is valid
    if lo > hi:
        return hi # Should not happen, but safe
    
    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
            
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    period = 27720 # LCM(1..12)
    
    best_m = 0
    best_f = -1
    
    # We only need to check m in [0, period-1]
    # Because f(12, m) = f(12, m % period)
    
    for m in range(period):
        if time.time() > deadline:
            break
            
        f_val = _compute_f(m)
        
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    # If we didn't finish, we return the best found so far.
    # Given the small period, we should finish well within time.
    
    return int(best_m)

# EVOLVE-BLOCK-END
