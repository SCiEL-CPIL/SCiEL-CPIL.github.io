import time
import math

def construct():
    start_time = time.time()
    limit = 85  # Leave 5s for overhead
    
    N = 12
    period = 27720
    max_f = 144
    
    def has_matching(m, L):
        # Precompute r_i for each i
        r = [0] * (N + 1)
        for i in range(1, N + 1):
            r[i] = (-m - 1) % i
        
        # Build candidate lists, early return if any i has no candidates
        cands = [None] * (N + 1)
        for i in range(1, N + 1):
            ri = r[i]
            if ri >= L:
                return False
            cands[i] = list(range(ri, L, i))
        
        # Order: fewest candidates first
        order = list(range(1, N + 1))
        order.sort(key=lambda i: len(cands[i]))
        
        # DFS-based matching
        match_to = [-1] * L
        
        def dfs(i, visited):
            for j in cands[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to[j] == -1 or dfs(match_to[j], visited):
                    match_to[j] = i
                    return True
            return False
        
        for i in order:
            visited = [False] * L
            if not dfs(i, visited):
                return False
        
        return True
    
    def compute_f(m):
        # Binary search on L in [12, 144]
        lo, hi = 12, max_f
        best = max_f
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best
    
    best_m = 0
    best_f = 0
    
    # Search through one full period
    for m in range(period):
        if m % 100 == 0 and time.time() - start_time > limit:
            break
        
        f_val = compute_f(m)
        
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f == max_f:
                break
    
    return best_m
