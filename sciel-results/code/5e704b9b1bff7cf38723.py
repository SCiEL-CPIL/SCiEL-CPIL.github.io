"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    
    def can_assign(m, L):
        """Check if there's a perfect matching: assign distinct b_i in (m, m+L] with i | b_i."""
        # For each index i in 1..12, find candidates in (m, m+L] divisible by i
        # Then check if perfect matching exists using DFS with backtracking
        # Since n=12 is small, use bitmask DP
        
        # Build adjacency: for each i, which positions (values) it can take
        # Values are m+1, m+2, ..., m+L
        # Map value to index: value v corresponds to position (v - m - 1) in 0..L-1
        
        # For bitmask DP: dp[mask] where mask represents which indices have been assigned
        # But we need to assign to specific values, so better: for each value, which indices can use it
        # Then do matching: for each value in order, try to assign to an unassigned index
        
        # Actually, let's use: for each index i, list of valid value-positions
        # Then check if there's a system of distinct representatives
        
        # Use DFS matching on bipartite graph: left = indices 0..11 (i=1..12), right = positions 0..L-1
        # Edge from i to pos if (m+1+pos) % (i+1) == 0
        
        # Since L can be up to ~144, and n=12, use DFS with memo or simple backtracking
        # Better: use Hopcroft-Karp or simple DFS matching (Kuhn's algorithm)
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            idx = i + 1  # actual index value
            # Find positions pos in 0..L-1 such that (m+1+pos) % idx == 0
            # m+1+pos ≡ 0 (mod idx) => pos ≡ -(m+1) (mod idx)
            start_val = m + 1
            # First multiple of idx >= m+1
            first = ((m + 1 + idx - 1) // idx) * idx
            if first > m + L:
                adj[i] = []
                continue
            # All multiples: first, first+idx, first+2*idx, ... <= m+L
            for v in range(first, m + L + 1, idx):
                pos = v - (m + 1)
                if 0 <= pos < L:
                    adj[i].append(pos)
        
        # Now check if perfect matching exists using DFS (Kuhn's algorithm)
        # match[pos] = which index is matched to position pos, or -1
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
        
        count = 0
        for i in range(n):
            visited = [False] * L
            if dfs(i, visited):
                count += 1
            else:
                return False
        return count == n
    
    def f_of_m(m):
        """Find minimum L such that can_assign(m, L) is True."""
        # L must be at least 12 (need 12 distinct values)
        # Upper bound: at most 144 per problem statement
        lo, hi = 12, 144
        if not can_assign(m, hi):
            return -1  # shouldn't happen for valid m
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Sample m values to find one with high f(12,m)
    # Strategy: try random samples and some structured ones
    best_m = 0
    best_f = -1
    
    # Try some structured candidates first
    candidates = []
    # Random samples
    random.seed(42)
    for _ in range(5000):
        candidates.append(random.randint(0, max_m))
    # Also try some specific values
    for v in range(0, max_m + 1, 1000):
        candidates.append(v)
    
    for m in candidates:
        if time.time() - start_time > deadline:
            break
        f_val = f_of_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
