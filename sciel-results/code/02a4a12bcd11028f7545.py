import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
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
            lst = get_candidates(m, L, i)
            if not lst:
                return False
            cands.append(lst)
        
        # Collect all unique values and map to indices
        all_vals = set()
        for i in range(N):
            for x in cands[i]:
                all_vals.add(x)
        
        if len(all_vals) < N:
            return False
        
        val_list = sorted(all_vals)
        val_to_idx = {v: idx for idx, v in enumerate(val_list)}
        num_vals = len(val_list)
        
        # Build adjacency: adj[i] = list of value-indices that divisor (i+1) can match
        adj = []
        for i in range(N):
            adj.append([val_to_idx[x] for x in cands[i]])
        
        # Kuhn's algorithm for bipartite matching
        matchR = [-1] * num_vals  # matchR[j] = left node i matched to right node j, or -1
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if matchR[j] == -1 or dfs(matchR[j], visited):
                    matchR[j] = i
                    return True
            return False
        
        result = 0
        for i in range(N):
            visited = [False] * num_vals
            if dfs(i, visited):
                result += 1
        
        return result == N
    
    def compute_f(m):
        """Find the minimum L >= 1 such that matching exists in (m, m+L]."""
        # Binary search on L since can_match is monotone in L
        lo = 1
        hi = 145
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Search m in one period
    # Use periodic time checks to ensure we don't exceed deadline
    check_interval = 500
    
    for m in range(period):
        if m % check_interval == 0:
            if time.time() > deadline:
                break
        
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)
