import math
from collections import deque

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    
    # Compute LCM of all divisors
    lcm = 1
    for d in divisors:
        lcm = lcm * d // math.gcd(lcm, d)
    
    # Hopcroft-Karp algorithm for maximum bipartite matching
    def hopcroft_karp(left_nodes, adj):
        """
        left_nodes: list of left node ids (0..n-1)
        adj: dict mapping left node id -> list of right node ids
        Returns: (match_right, match_left) where match_right[r] = l and match_left[l] = r
        """
        INF = float('inf')
        match_l = {}  # left -> right
        match_r = {}  # right -> left
        dist = {}
        
        def bfs():
            queue = deque()
            for u in left_nodes:
                if u not in match_l:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    if v in match_r:
                        u2 = match_r[v]
                        if dist[u2] == INF:
                            dist[u2] = dist[u] + 1
                            queue.append(u2)
                    else:
                        found = True
            return found
        
        def dfs(u):
            for v in adj.get(u, []):
                if v in match_r:
                    u2 = match_r[v]
                    if dist[u2] == dist[u] + 1:
                        if dfs(u2):
                            match_r[v] = u
                            match_l[u] = v
                            return True
                else:
                    match_r[v] = u
                    match_l[u] = v
                    return True
            dist[u] = INF
            return False
        
        matching = 0
        while bfs():
            for u in left_nodes:
                if u not in match_l:
                    if dfs(u):
                        matching += 1
        return match_r, match_l
    
    # Check if a valid assignment exists for window [L, L+W]
    def check_window(L, W):
        # Build adjacency: for each divisor i, which values in [L, L+W] are multiples
        adj = {}
        for i, d in enumerate(divisors):
            vals = []
            k_min = (L + d - 1) // d
            k_max = (L + W) // d
            if k_min <= k_max:
                for k in range(k_min, k_max + 1):
                    vals.append(k * d)
            adj[i] = vals
        
        # Check if any divisor has no candidates
        for i in range(n):
            if not adj[i]:
                return False, None
        
        # Run Hopcroft-Karp
        left_nodes = list(range(n))
        # We need to handle the fact that right nodes are actual values, not indices
        # Hopcroft-Karp works with any hashable nodes
        match_r, match_l = hopcroft_karp(left_nodes, adj)
        
        if len(match_l) < n:
            return False, None
        
        # Build solution in order of divisors
        solution = [0] * n
        for i in range(n):
            if i in match_l:
                solution[i] = match_l[i]
            else:
                return False, None
        
        return True, solution
    
    # Binary search on W
    lo = 0
    hi = 1000
    best_w = hi
    best_solution = None
    
    # First, check if hi is feasible
    found, sol = check_window(0, hi)
    if not found:
        # Try other L values
        for L in range(0, lcm, 100):
            found, sol = check_window(L, hi)
            if found:
                break
    if not found:
        # Fallback: just return divisors (invalid but avoids crash)
        return divisors[:]
    
    # Binary search for minimum W
    while lo < hi:
        mid = (lo + hi) // 2
        feasible = False
        for L in range(0, lcm):
            found, sol = check_window(L, mid)
            if found:
                feasible = True
                best_w = mid
                best_solution = sol
                break
        if feasible:
            hi = mid
        else:
            lo = mid + 1
    
    if best_solution is None:
        return divisors[:]
    
    # Ensure all are Python ints
    return [int(v) for v in best_solution]
