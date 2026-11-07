import time

def construct():
    start_time = time.time()
    limit = 85  # safety margin
    
    N = 12
    period = 27720  # lcm(1..12)
    
    best_m = 0
    best_f = 0
    
    def get_candidates(m, L, i):
        """Return list of x in (m, m+L] divisible by i"""
        start = m + 1
        rem = start % i
        if rem == 0:
            first = start
        else:
            first = start + (i - rem)
        
        res = []
        for x in range(first, m + L + 1, i):
            res.append(x)
        return res
    
    def can_match(m, L):
        # Build candidate lists for i=1..12
        cands = []
        for i in range(1, N + 1):
            cands.append(get_candidates(m, L, i))
        
        # Check if any i has no candidates
        for i in range(N):
            if len(cands[i]) == 0:
                return False
        
        # Build a mapping from value to which i's can use it
        # Then use bipartite matching
        # Left side: i = 0..11 (representing 1..12)
        # Right side: values
        
        # For Kuhn's algorithm, we need to find a matching
        # match[i] = value matched to i, or -1
        match = [-1] * N
        
        def dfs(i, visited):
            for x in cands[i]:
                if visited[x]:
                    continue
                visited[x] = True
                # Find which j has x as candidate, if any
                # Actually, we need to check if x is matched to some j
                # and try to reassign j
                # This is tricky because multiple i's can have the same x
                pass
        
        # Let me use a different approach: build adjacency list
        # adj[i] = list of values that i can match to
        # Then use standard bipartite matching
        
        # For efficiency, let's use a set for visited values
        # and try to find augmenting paths
        
        # Actually, for 12 nodes, let's just use the backtracking but with better ordering
        indices = list(range(N))
        indices.sort(key=lambda idx: len(cands[idx]))
        
        used = set()
        
        def backtrack(k):
            if k == N:
                return True
            i = indices[k]
            for x in cands[i]:
                if x not in used:
                    used.add(x)
                    if backtrack(k + 1):
                        return True
                    used.remove(x)
            return False
        
        return backtrack(0)
    
    def compute_f(m):
        # Binary search on L
        lo = 1
        hi = 200
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Search m in one period
    check_interval = 500
    
    for m in range(period):
        if m % check_interval == 0:
            if time.time() - start_time > limit:
                break
        
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return best_m
