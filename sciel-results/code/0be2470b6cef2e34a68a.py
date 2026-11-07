import time
import math

def construct():
    start_time = time.time()
    limit = 85  # Leave 5s for overhead
    
    N = 12
    period = 27720
    
    # Precompute candidate positions for each i and residue class
    # For a given m, the window is (m, m+L] = {m+1, ..., m+L}
    # We need to assign distinct positions j in 0..L-1 to each i in 1..12
    # such that (m+1+j) % i == 0, i.e., j ≡ (-m-1) % i (mod i)
    
    # Strategy:
    # 1. For each m in 0..period-1, compute f(m) = min L such that matching exists.
    # 2. To speed up, note that f(m) depends only on m mod lcm(1..12) = 27720.
    # 3. Use a faster matching algorithm or pruning.
    
    # Precompute for each i in 1..12 the list of residues r such that r % i == 0
    # Actually, for a fixed m and L, position j is valid for i iff (m+1+j) % i == 0.
    # Let r_i = (-m-1) % i. Then j ≡ r_i (mod i).
    # So for each i, valid j are r_i, r_i+i, r_i+2i, ... < L.
    
    def compute_f(m):
        # Compute the least L >= 1 such that there's a perfect matching
        # Start with L=1, increase until found
        # Upper bound: by pigeonhole / density arguments, L is at most ~100
        # But to be safe, use 150
        
        # Precompute r_i for each i
        r = [0] * (N + 1)
        for i in range(1, N + 1):
            r[i] = (-m - 1) % i
        
        for L in range(1, 151):
            # Build candidate lists
            # cands[i] = list of j in 0..L-1 such that j % i == r[i]
            cands = [[] for _ in range(N + 1)]
            for i in range(1, N + 1):
                ri = r[i]
                if ri < L:
                    cands[i] = list(range(ri, L, i))
            
            # Check if perfect matching exists using Hopcroft-Karp or simple DFS
            # Since N=12 is small, DFS (Kuhn) is fast enough
            match_to = [-1] * L  # match_to[j] = i or -1
            
            def dfs(i, visited):
                for j in cands[i]:
                    if visited[j]:
                        continue
                    visited[j] = True
                    if match_to[j] == -1 or dfs(match_to[j], visited):
                        match_to[j] = i
                        return True
                return False
            
            # Try to match all i from 1 to 12
            # Order matters for performance: try harder constraints first (larger i)
            # Actually, larger i have fewer candidates, so try them first
            order = list(range(1, N + 1))
            order.sort(key=lambda i: len(cands[i]))  # Fewest candidates first
            
            success = True
            for i in order:
                visited = [False] * L
                if not dfs(i, visited):
                    success = False
                    break
            
            if success:
                return L
        
        return 150  # Should not happen
    
    best_m = 0
    best_f = 0
    
    # Search through one full period
    # To speed up, we can sample or use smarter search
    # But since period is only 27720 and each f(m) is fast, let's try all
    
    # Optimization: cache results? No, each m is different.
    
    # Let's iterate and track time
    for m in range(period):
        if time.time() - start_time > limit:
            break
        
        f_val = compute_f(m)
        
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return best_m
