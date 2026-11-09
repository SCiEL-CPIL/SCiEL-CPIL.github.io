import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill

    P = 27720  # LCM(1..12)
    n = 12
    max_L = 144  # upper bound from contract

    # For each residue r in 0..P-1, compute f(12, r)
    # f(12, r) = smallest L >= 1 such that matching exists in (r, r+L]
    # Actually, we want to MAXIMIZE f, so we want the LARGEST L where matching STILL exists
    # Wait: f(12,m) is the LEAST positive L such that (m, m+L] contains the distinct b_i.
    # So f is the minimum L that works. To maximize f, we want residues where the minimum
    # working L is as large as possible.
    
    # For a given r, f(12,r) = min L >= 1 such that matching exists in window (r, r+L]
    # The window (r, r+L] contains integers r+1, r+2, ..., r+L
    # We need to assign to each i in 1..12 a distinct value b_i in {r+1,...,r+L} with i|b_i
    
    # For each r, we can binary search or just iterate L from 1 upward to find the first L that works.
    # But we want to find r that MAXIMIZES this minimum L.
    
    # Let's precompute: for each r, what is the smallest L that works?
    # Then return r that gives the largest such L.
    
    # To speed up matching: use DFS with ordering by decreasing i (i=12 first, most constrained)
    
    def has_matching(r, L):
        """Check if there's a matching for residue r with window size L."""
        # Left nodes: i = 1..12 (we'll process in order 12,11,...,1)
        # Right nodes: positions 0..L-1 (representing r+1+j)
        # Edge i -> j if (r + 1 + j) % i == 0
        
        # For speed, precompute for each i which j's are valid
        valid = [None] * (n + 1)
        for i in range(1, n + 1):
            valid[i] = []
            for j in range(L):
                if (r + 1 + j) % i == 0:
                    valid[i].append(j)
        
        # If any i has no valid j, return False
        for i in range(1, n + 1):
            if len(valid[i]) < 1:
                return False
        
        # DFS matching, process i from 12 down to 1 (most constrained first)
        match_to = [-1] * L  # match_to[j] = i that is matched to position j, or -1
        
        def dfs(i, visited):
            for j in valid[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to[j] == -1 or dfs(match_to[j], visited):
                    match_to[j] = i
                    return True
            return False
        
        # Try to match all i from 12 down to 1
        for i in range(n, 0, -1):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        
        return True
    
    # For each residue r, find f(12, r) = smallest L >= 1 with matching
    # We want to maximize this value
    best_L = 0
    best_r = 0
    
    for r in range(P):
        # Check if we're still within deadline
        if time.time() > deadline:
            break
        
        # Find smallest L >= 1 such that has_matching(r, L) is True
        # L must be at least 12 (need 12 distinct numbers)
        found = False
        for L in range(12, max_L + 1):
            if has_matching(r, L):
                # This is the smallest L that works for this r
                if L > best_L:
                    best_L = L
                    best_r = r
                found = True
                break
        
        # If no L up to max_L works, this r is invalid (shouldn't happen for large enough L)
    
    # Return best_r as m (0 <= m <= 1000000)
    # best_r is in 0..P-1 = 0..27719, which is within 0..1000000
    return int(best_r)
