"""Find worst interval start for distinct-multiple matching with n=12."""
import time
from math import gcd
from collections import deque

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_lcm_table(n):
    L = 1
    for i in range(2, n + 1):
        L = _lcm(L, i)
    return L

def _hopcroft_karp(adj, n_left):
    """
    Hopcroft-Karp algorithm for maximum bipartite matching.
    adj: dict mapping left node -> list of right nodes
    n_left: number of left nodes (1..n_left)
    Returns size of maximum matching.
    """
    match_left = {}  # left -> right
    match_right = {}  # right -> left
    dist = {}
    
    def bfs():
        queue = deque()
        for u in range(1, n_left + 1):
            if u not in match_left:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        found = False
        while queue:
            u = queue.popleft()
            for v in adj.get(u, []):
                if v not in match_right:
                    found = True
                elif dist.get(match_right[v], float('inf')) == float('inf'):
                    dist[match_right[v]] = dist[u] + 1
                    queue.append(match_right[v])
        return found
    
    def dfs(u):
        for v in adj.get(u, []):
            if v not in match_right or (dist.get(match_right[v], float('inf')) == dist.get(u, 0) + 1 and dfs(match_right[v])):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = float('inf')
        return False
    
    matching = 0
    while bfs():
        for u in range(1, n_left + 1):
            if u not in match_left and dfs(u):
                matching += 1
    return matching

def _can_match(m, L, n):
    """Check if (m, m+L] contains distinct b_i with i | b_i for i=1..n."""
    adj = {}
    for i in range(1, n + 1):
        first = (m // i + 1) * i
        last = ((m + L) // i) * i
        if first > last:
            return False
        adj[i] = list(range(first, last + 1, i))
    
    # Quick check: each i must have at least one candidate
    for i in range(1, n + 1):
        if not adj[i]:
            return False
    
    # Use Hopcroft-Karp
    matching_size = _hopcroft_karp(adj, n)
    return matching_size == n

def _compute_f(m, n, max_L=200):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    for L in range(1, max_L + 1):
        if _can_match(m, L, n):
            return L
    return max_L

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin (evaluator kills at 90s)
    
    n = 12
    LCM = _build_lcm_table(n)  # 27720
    
    best_m = 0
    best_f = 0
    
    # Precompute f(m) for all residue classes mod LCM
    # f(m) depends only on m mod LCM because divisibility conditions repeat with period LCM
    for r in range(LCM):
        if time.time() > deadline:
            break
        f_val = _compute_f(r, n)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    # best_m is a residue in 0..27719, so it's always <= 1000000
    result = best_m
    
    return int(result)

construct()
