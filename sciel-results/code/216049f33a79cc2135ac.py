"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 12, 'maximum_start': 1000000}

def _has_perfect_matching(m, L):
    """Check if there exist distinct b_1..b_12 in (m, m+L] with i|b_i."""
    n = 12
    adj = [[] for _ in range(n + 1)]  # 1-indexed, adj[i] = list of numbers in (m,m+L] div by i
    
    for i in range(1, n + 1):
        first = ((m + 1 + i - 1) // i) * i
        if first <= m + L:
            for num in range(first, m + L + 1, i):
                adj[i].append(num)
        else:
            return False  # no candidate for i
    
    # Bipartite matching via DFS augmenting paths
    match_right = {}
    
    def dfs(i, visited):
        for num in adj[i]:
            if num in visited:
                continue
            visited.add(num)
            if num not in match_right or dfs(match_right[num], visited):
                match_right[num] = i
                return True
        return False
    
    for i in range(1, n + 1):
        visited = set()
        if not dfs(i, visited):
            return False
    
    return True

def _compute_f(m):
    """Compute f(12, m): least L >= 12 such that (m, m+L] has distinct b_i with i|b_i."""
    n = 12
    for L in range(n, 200):
        if _has_perfect_matching(m, L):
            return L
    return 200

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_m = 0
    best_L = 0
    
    max_m = P['maximum_start']
    
    for m in range(max_m + 1):
        if time.time() > deadline:
            break
        L = _compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
