"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def _has_matching(m, L):
    """
    Check if interval (m, m+L] has distinct b_1..b_12 with i|b_i.
    Left nodes: 1..12
    Right nodes: 0..L-1 (representing m+1 .. m+L)
    Edge i -> j if i divides (m+1+j)
    """
    n_left = 12
    
    # Build adjacency list
    # adj[i-1] contains indices j in 0..L-1 such that i | (m+1+j)
    adj = [[] for _ in range(n_left)]
    
    # Precompute m+1 modulo i for each i to speed up generation?
    # Or just iterate.
    for i in range(1, n_left + 1):
        # We need j such that (m + 1 + j) % i == 0
        # j % i == (-(m+1)) % i
        target_res = (-(m + 1)) % i
        
        # Find first j in [0, L-1] with j % i == target_res
        # j = target_res, target_res + i, ...
        # If target_res >= L, no candidates
        if target_res < L:
            j = target_res
            while j < L:
                adj[i-1].append(j)
                j += i
        
        # If no candidates, matching is impossible
        if not adj[i-1]:
            return False

    # Kuhn's algorithm with heuristic ordering
    # Order left nodes by degree (fewest options first)
    # This helps DFS prune earlier
    order = sorted(range(n_left), key=lambda u: len(adj[u]))
    
    # match_r[v] = u means right node v is matched to left node u
    match_r = [-1] * L
    
    # DFS function
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    # We need to map the reordered left nodes back to original logic?
    # Actually, Kuhn's algorithm just needs to find a matching.
    # The "left" nodes are abstract. We can just process them in the order 'order'.
    # But 'match_r' maps right node to left node index.
    # If we process left node 'u' (which is an index 0..11), it's fine.
    
    for u in order:
        visited = [False] * L
        if not dfs(u, visited):
            return False
            
    return True

def _compute_f(m):
    """
    Compute f(12, m): least L such that (m, m+L] has valid assignment.
    """
    # Lower bound: max distance to next multiple for any i
    lo = 1
    for i in range(1, 13):
        r = m % i
        if r == 0:
            dist = i
        else:
            dist = i - r
        if dist > lo:
            lo = dist
            
    hi = 144 # Max possible L per task description
    
    if lo > hi:
        return hi
        
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
            
    return int(best_m)

# EVOLVE-BLOCK-END
