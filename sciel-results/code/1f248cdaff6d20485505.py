import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 12
    P = 27720  # lcm(1..12)
    UB = 144   # Max possible L per contract
    
    def has_matching(m, L):
        """
        Check if there is a perfect matching in the bipartite graph
        Left: divisors i in 1..12
        Right: integers b in (m, m+L]
        Edge (i, b) exists if i divides b.
        """
        # Build adjacency list for left nodes (0..11 representing i=1..12)
        # We map right nodes (integers b) to indices 0..R-1 for efficient matching arrays
        adj = [[] for _ in range(n)]
        b_to_idx = {}
        idx = 0
        
        for i in range(1, n + 1):
            # First multiple of i strictly greater than m
            # ceil((m+1)/i) * i
            # (m // i + 1) * i is equivalent for integer m
            first = (m // i + 1) * i
            end_val = m + L
            if first > end_val:
                # No candidates for this i, matching impossible
                # But we must check if ALL i have candidates? 
                # If one i has no candidates, return False immediately? 
                # Yes, if any left node has degree 0, no perfect matching.
                # However, we build the list first. If list is empty, dfs will fail.
                pass
            
            lst = []
            # Iterate multiples of i in (m, m+L]
            # Step is i
            for v in range(first, end_val + 1, i):
                if v not in b_to_idx:
                    b_to_idx[v] = idx
                    idx += 1
                lst.append(b_to_idx[v])
            adj[i - 1] = lst
            
            # Early exit: if any i has no candidates, we can't match
            if not lst:
                return False
        
        # If total candidates idx is less than n, impossible
        if idx < n:
            return False
            
        # Kuhn's Algorithm (DFS based augmenting path)
        match_right = [-1] * idx  # match_right[v_idx] = u_idx (left node)
        
        # Optimization: Order left nodes by degree (ascending) to prune earlier?
        # For n=12, overhead of sorting might not be worth it, but let's try simple order first.
        # Actually, standard Kuhn is fast enough.
        
        def dfs(u, seen):
            for v in adj[u]:
                if seen[v]:
                    continue
                seen[v] = True
                if match_right[v] == -1 or dfs(match_right[v], seen):
                    match_right[v] = u
                    return True
            return False
            
        # Try to match all left nodes
        for u in range(n):
            seen = [False] * idx
            if not dfs(u, seen):
                return False
        return True

    def f_for_m(m):
        """
        Find the least L such that has_matching(m, L) is True.
        Uses binary search since has_matching is monotonic with L.
        """
        # Check if even UB is not enough (should not happen per contract, but safe)
        if not has_matching(m, UB):
            return UB
        
        lo, hi = 1, UB
        ans = UB
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans

    best_m = 0
    best_f = -1
    
    # We search all residues modulo P
    # To maximize speed, we can check time periodically
    # 27720 iterations. If each takes ~0.1ms, total ~3s.
    # If each takes ~1ms, total ~30s.
    # Binary search reduces L checks from ~144 to ~8.
    
    for m in range(P):
        if time.time() > deadline:
            break
        
        val = f_for_m(m)
        if val > best_f:
            best_f = val
            best_m = m
            
    return int(best_m)
