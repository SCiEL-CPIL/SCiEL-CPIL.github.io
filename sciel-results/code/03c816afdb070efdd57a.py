"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from functools import lru_cache

N = 12
MAX_M = 1000000
MAX_L = 144
P = 27720  # lcm(1..12)

def construct():
    deadline = time.time() + 85  # 85s safety margin for 90s limit
    
    # Precompute masks for one period
    # mask[d] for d in 1..P, where bit i-1 is set if i divides d
    masks = [0] * (P + 1)
    for d in range(1, P + 1):
        m = 0
        for i in range(1, N + 1):
            if d % i == 0:
                m |= (1 << (i - 1))
        masks[d] = m

    best_r = 0
    best_f = 0

    # Function to check if a perfect matching exists for a given window (r, r+L]
    # We check if there is a bijection between indices 1..N and distinct numbers in (r, r+L]
    # such that i divides the assigned number.
    def check_matching(r, L):
        # adj[i] = list of indices j (0..L-1) such that (i+1) divides (r+1+j)
        adj = [[] for _ in range(N)]
        for j in range(L):
            x = r + 1 + j
            xm = x % P
            if xm == 0:
                mx = masks[P]
            else:
                mx = masks[xm]
            
            for i in range(N):
                if mx & (1 << i):
                    adj[i].append(j)
        
        # DFS matching with bitmasks for speed
        match_to_num = [-1] * L
        match_to_idx = [-1] * N
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to_num[j] == -1 or dfs(match_to_num[j], visited):
                    match_to_num[j] = i
                    match_to_idx[i] = j
                    return True
            return False
        
        for i in range(N):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        
        return True

    # For each residue r, find the minimum L such that matching exists in (r, r+L]
    # Use binary search since the property is monotone: if matching exists for L, it exists for L+1
    
    def min_L_for_residue(r):
        # Binary search for smallest L in [1, MAX_L] where check_matching(r, L) is True
        lo, hi = 1, MAX_L
        while lo < hi:
            mid = (lo + hi) // 2
            if check_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1
        # lo is the answer if check_matching(r, lo) is True, else no matching in [1, MAX_L]
        if check_matching(r, lo):
            return lo
        return -1  # No matching found

    # Iterate over all residues, but with early termination and smarter ordering
    # Try residues that are likely to have larger f values first
    # Heuristic: residues where the first few positions have few divisors
    
    # Precompute a heuristic score for each residue to prioritize
    # A residue r is "hard" if positions r+1, r+2, ... have few divisors among 1..12
    
    residue_order = list(range(P))
    
    # Compute heuristic: count of "easy" positions (positions with many divisors)
    # in the first MAX_L positions after r
    # Lower count = harder = more likely to have larger f
    def hardness_score(r):
        count = 0
        for j in range(1, min(MAX_L, P - r) + 1):
            x = r + j
            xm = x % P
            if xm == 0:
                mx = masks[P]
            else:
                mx = masks[xm]
            count += bin(mx).count('1')
        return count
    
    # Sort by hardness (ascending = harder first)
    # But this is expensive, so let's just iterate and use early termination
    
    for r in range(P):
        if time.time() > deadline:
            break
        
        # Quick upper bound: if best_f == MAX_L, we can't do better
        if best_f >= MAX_L:
            break
        
        current_f = min_L_for_residue(r)
        
        if current_f > best_f:
            best_f = current_f
            best_r = r
    
    # The best m is best_r (since best_r < P <= MAX_M)
    return int(best_r)

# EVOLVE-BLOCK-END
