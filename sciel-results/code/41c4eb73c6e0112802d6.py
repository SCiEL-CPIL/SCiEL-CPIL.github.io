import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill
    
    N = 12
    period = 27720  # LCM(1,2,...,12)
    MAX_L = 150
    
    def has_matching(m, L):
        """Check if perfect matching exists for interval (m, m+L]."""
        # For each i in 1..12, find valid positions j in [1, L] where (m+j) % i == 0
        # j ≡ r_i (mod i) where r_i = (-m-1) % i, and 1 <= j <= L
        cands = [None] * (N + 1)
        for i in range(1, N + 1):
            r = (-m - 1) % i
            # j values: r, r+i, r+2i, ... but j must be >= 1
            # First valid j: if r == 0 then j = i, else j = r (if r >= 1)
            # Actually j must be in [1, L]
            # j ≡ r (mod i), j >= 1, j <= L
            # Smallest j >= 1 with j ≡ r (mod i):
            #   if r >= 1: j_start = r
            #   if r == 0: j_start = i
            if r == 0:
                j_start = i
            else:
                j_start = r
            
            if j_start > L:
                return False  # No candidates for i
            
            cands[i] = list(range(j_start, L + 1, i))
        
        # Quick check: all must have at least one candidate
        for i in range(1, N + 1):
            if not cands[i]:
                return False
        
        # Hopcroft-Karp for bipartite matching
        # Left side: i in 1..12, Right side: j in 1..L
        # Build adjacency: adj[i] = list of j values
        adj = [None] + cands[1:]  # 1-indexed
        
        # Hopcroft-Karp
        # pair_u[i] = matched j for i, -1 if unmatched
        # pair_v[j] = matched i for j, -1 if unmatched
        pair_u = [-1] * (N + 1)
        pair_v = [-1] * (L + 1)
        
        from collections import deque
        
        def bfs():
            """BFS to find augmenting paths, returns distance labels."""
            dist = [-1] * (N + 1)
            queue = deque()
            for i in range(1, N + 1):
                if pair_u[i] == -1:
                    dist[i] = 0
                    queue.append(i)
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return dist, found
        
        def dfs(u, dist):
            """DFS for augmenting path from u."""
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu, dist)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1  # Prune
            return False
        
        matching = 0
        while True:
            dist, found = bfs()
            if not found:
                break
            for i in range(1, N + 1):
                if pair_u[i] == -1:
                    if dfs(i, dist):
                        matching += 1
        
        return matching == N
    
    def compute_f(m):
        """Compute f(12, m) = smallest L >= 1 with perfect matching."""
        # Binary search on L
        lo, hi = 1, MAX_L
        # First check if even MAX_L works
        if not has_matching(m, MAX_L):
            return MAX_L  # Should not happen for reasonable L
        
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Precompute f(m) for all m in [0, period)
    # Optimize: note that f(m) depends on m mod lcm(1..12) = 27720
    # So we just need to find the best m in [0, 27720)
    
    best_m = 0
    best_f = 0
    
    # Full sweep over all residues
    for m in range(period):
        if time.time() - start_time > DEADLINE:
            break
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return best_m
