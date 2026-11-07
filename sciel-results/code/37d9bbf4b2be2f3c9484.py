import sys
import time

def construct():
    N = 12
    MOD = 27720
    best_m = 0
    best_L = 0
    
    # Deadline guard: evaluator kills at 90s wall-clock.
    # Use 85s as safety margin.
    deadline = time.time() + 85.0
    
    # Precompute for each i in 1..N and each residue class r = m % i,
    # the sorted list of offsets d in 1..max_L_search such that (m+d) % i == 0.
    # (m+d) % i == 0  <=>  d % i == (-m) % i.
    # So for a given m, for each i, the valid offsets are those d where
    # d % i == (-m) % i.
    
    max_L_search = 200  # Safe upper bound for f(12, m)
    
    # Precompute: for each i (1..N), for each residue r (0..i-1),
    # list of d in 1..max_L_search with d % i == r.
    offsets_by_residue = {}
    for i in range(1, N + 1):
        offsets_by_residue[i] = []
        for r in range(i):
            lst = []
            d = r
            if d == 0:
                d = i  # d must be >= 1
            while d <= max_L_search:
                lst.append(d)
                d += i
            offsets_by_residue[i].append(lst)
    
    for m in range(MOD):
        if time.time() > deadline - 5.0:
            break
        
        # For this m, find minimal L
        found_L = None
        for L in range(1, max_L_search + 1):
            if check_matching_fast(m, L, N, offsets_by_residue):
                found_L = L
                break
        
        if found_L is not None:
            if found_L > best_L:
                best_L = found_L
                best_m = m
    
    return int(best_m)

def check_matching_fast(m, L, N, offsets_by_residue):
    # Build adjacency using precomputed offsets.
    # For each index i, the numbers in (m, m+L] divisible by i are
    # m + d where d is in offsets_by_residue[i][(-m) % i] and d <= L.
    
    adj = [[] for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        r = (-m) % i
        valid_offsets = offsets_by_residue[i][r]
        # Filter to d <= L
        # Since valid_offsets is sorted, we can binary search or just filter.
        # For small L, filtering is fine.
        for d in valid_offsets:
            if d > L:
                break
            adj[i].append(m + d)
    
    # Bipartite matching: left side = indices 1..N, right side = numbers.
    # Use Kuhn's algorithm (DFS-based augmenting paths).
    
    match_to_idx = {}  # num -> idx
    
    def dfs(idx, visited):
        for num in adj[idx]:
            if num in visited:
                continue
            visited.add(num)
            if num not in match_to_idx or dfs(match_to_idx[num], visited):
                match_to_idx[num] = idx
                return True
        return False
    
    # Order indices by size of adj list (most constrained first)
    indices = list(range(1, N + 1))
    indices.sort(key=lambda i: len(adj[i]))
    
    for idx in indices:
        if not dfs(idx, set()):
            return False
    
    return True
