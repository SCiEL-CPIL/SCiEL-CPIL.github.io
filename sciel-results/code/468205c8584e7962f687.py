import time
from math import gcd
from functools import reduce

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
PERIOD = reduce(lambda a, b: a * b // gcd(a, b), DIVISORS)  # 60060

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = N
    divisors = DIVISORS
    max_val = MAX_VAL
    period = PERIOD

    def has_perfect_matching(adj, n):
        """Check if bipartite graph has perfect matching using DFS-based augmenting paths.
        adj[i] is list of j's that node i can match to. n is number of nodes on each side."""
        match_r = [-1] * n  # match_r[j] = i means right node j matched to left node i
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_r[v] == -1 or dfs(match_r[v], visited):
                    match_r[v] = u
                    return True
            return False
        
        for u in range(n):
            visited = [False] * n
            if not dfs(u, visited):
                return False
        return True

    def find_assignment_matching(L, W):
        """Find assignment in window [L, L+W) using bipartite matching."""
        if W < n:
            return None
        if L + W > max_val:
            return None

        # Build adjacency: for each divisor i, which positions j (0..W-1) are valid
        adj = []
        for i in range(n):
            d = divisors[i]
            rem = (-L) % d
            if rem >= W:
                return None
            positions = list(range(rem, W, d))
            adj.append(positions)
        
        # Check perfect matching
        if not has_perfect_matching(adj, n):
            return None
        
        # Extract the matching
        match_r = [-1] * n
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_r[v] == -1 or dfs(match_r[v], visited):
                    match_r[v] = u
                    return True
            return False
        
        for u in range(n):
            visited = [False] * n
            dfs(u, visited)
        
        # Build result
        result = [0] * n
        for v in range(n):
            u = match_r[v]
            result[u] = L + v
        
        # Verify
        for i in range(n):
            if result[i] % divisors[i] != 0 or not (1 <= result[i] <= max_val):
                return None
        # Check distinctness
        if len(set(result)) != n:
            return None
        
        return result

    best_diff = float('inf')
    best_result = None

    # Phase 1: Try small W values
    for W in range(n, n + 25):
        if time.time() > deadline - 30:
            break
        for L in range(1, min(period + 1, max_val - W + 2)):
            if time.time() > deadline - 30:
                break
            # Quick feasibility: each divisor must have at least one multiple in [L, L+W-1]
            feasible = True
            for i in range(n):
                d = divisors[i]
                rem = (-L) % d
                if rem >= W:
                    feasible = False
                    break
            if not feasible:
                continue
            
            result = find_assignment_matching(L, W)
            if result:
                if W < best_diff:
                    best_diff = W
                    best_result = result
                break  # Found for this W, try next W

    # Phase 2: If found, try smaller W
    if best_diff < float('inf'):
        for W in range(n, best_diff):
            if time.time() > deadline - 15:
                break
            for L in range(1, min(period + 1, max_val - W + 2)):
                if time.time() > deadline - 15:
                    break
                feasible = True
                for i in range(n):
                    d = divisors[i]
                    rem = (-L) % d
                    if rem >= W:
                        feasible = False
                        break
                if not feasible:
                    continue
                
                result = find_assignment_matching(L, W)
                if result:
                    if W < best_diff:
                        best_diff = W
                        best_result = result
                    break

    # Phase 3: Deeper search with binary search on W for promising L values
    if time.time() < deadline - 5 and best_diff > n:
        for L in range(1, min(period + 1, max_val - n + 2)):
            if time.time() > deadline - 3:
                break
            
            hi = min(best_diff - 1, max_val - L)
            lo = n - 1
            
            if lo > hi:
                continue
            
            found_width = None
            while lo <= hi:
                if time.time() > deadline - 3:
                    break
                mid = (lo + hi) // 2
                # Quick feasibility
                feasible = True
                for i in range(n):
                    d = divisors[i]
                    rem = (-L) % d
                    if rem >= mid:
                        feasible = False
                        break
                if feasible:
                    result = find_assignment_matching(L, mid)
                    if result:
                        found_width = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1
                else:
                    lo = mid + 1
            
            if found_width is not None and found_width < best_diff:
                best_diff = found_width
                best_result = find_assignment_matching(L, found_width)

    if best_result:
        return best_result
    
    return [divisors[i] for i in range(n)]
