"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    LCM_12 = 27720  # LCM(1,2,...,12)
    
    def can_assign(m, L):
        """Check if there's a perfect matching using Kuhn's algorithm."""
        adj = [[] for _ in range(n)]
        for i in range(n):
            idx = i + 1
            first = ((m + idx) // idx) * idx
            if first > m + L:
                adj[i] = []
                continue
            for v in range(first, m + L + 1, idx):
                pos = v - m - 1
                adj[i].append(pos)
        
        for i in range(n):
            if not adj[i]:
                return False
        
        match = [-1] * L
        
        def dfs(i, visited):
            for pos in adj[i]:
                if visited[pos]:
                    continue
                visited[pos] = True
                if match[pos] == -1 or dfs(match[pos], visited):
                    match[pos] = i
                    return True
            return False
        
        for i in range(n):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True
    
    def f_of_m(m):
        """Find minimum L such that can_assign(m, L) is True."""
        lo, hi = 12, 144
        if not can_assign(m, hi):
            return 0
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Key insight: f(12, m) depends only on m mod LCM_12.
    # Exhaustively search all 27720 residue classes to find the global optimum.
    best_m = 0
    best_f = 0
    
    for r in range(LCM_12):
        if time.time() - start_time > deadline:
            break
        f_val = f_of_m(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r
            if best_f >= 144:
                break
    
    # Ensure best_m is within [0, max_m]
    # Since r < LCM_12 <= max_m, best_m is already valid
    return int(best_m)
# EVOLVE-BLOCK-END
