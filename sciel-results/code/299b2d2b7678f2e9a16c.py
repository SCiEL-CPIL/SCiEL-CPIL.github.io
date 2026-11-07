"""Cochromatic partition via branch-and-bound over maximal homogeneous sets."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 55.0
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full = (1 << n) - 1
    N = 1 << n
    
    # Precompute for each subset: is it a clique? independent set?
    is_clique = [False] * N
    is_indep = [False] * N
    is_clique[0] = True
    is_indep[0] = True
    
    for S in range(1, N):
        lb = S & (-S)
        v = lb.bit_length() - 1
        rest = S ^ lb
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[S] = True
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[S] = True
    
    # valid[S] = is_clique[S] or is_indep[S], and S nonempty
    valid = [False] * N
    for S in range(1, N):
        valid[S] = is_clique[S] or is_indep[S]
    
    # Find all maximal homogeneous sets (maximal cliques and maximal independent sets)
    # A set is maximal if no valid superset exists
    maximal_sets = []
    for S in range(1, N):
        if valid[S]:
            # Check if S is maximal: no valid superset of S exists
            is_maximal = True
            # Try adding each vertex not in S
            for v in range(n):
                if not (S >> v) & 1:
                    if valid[S | (1 << v)]:
                        is_maximal = False
                        break
            if is_maximal:
                maximal_sets.append(S)
    
    # Sort maximal sets by size descending (larger sets first for better pruning)
    maximal_sets.sort(key=lambda x: x.bit_count(), reverse=True)
    
    # Branch-and-bound to find minimum number of disjoint homogeneous sets covering all vertices
    # We search for k = 1, 2, 3, ... and stop at the first feasible cover
    
    best_solution = None
    best_k = n + 1
    
    # Precompute: for each maximal set, which vertices it covers
    # And for quick checking, we'll use bitmask operations
    
    def branch_and_bound(k, remaining, chosen, depth):
        """Try to cover 'remaining' vertices with at most 'k' more homogeneous sets."""
        nonlocal best_solution, best_k
        
        if time.time() > deadline:
            return False
        
        if k == 0:
            return remaining == 0
        
        if remaining == 0:
            if depth < best_k:
                best_k = depth
                best_solution = list(chosen)
            return True
        
        # Lower bound: we need at least ceil(popcount(remaining) / max_set_size) more sets
        # But a simpler bound: if k < 1 and remaining != 0, fail
        if k < 1:
            return False
        
        # If even taking the largest possible set from remaining doesn't help enough
        # Try each maximal set that is a subset of remaining
        found = False
        for ms in maximal_sets:
            if ms & ~remaining:
                continue  # ms is not a subset of remaining
            
            # Choose this maximal set
            new_remaining = remaining ^ ms
            chosen.append(ms)
            
            if branch_and_bound(k - 1, new_remaining, chosen, depth + 1):
                found = True
                chosen.pop()
                return True
            
            chosen.pop()
            
            # Pruning: if we've tried enough and time is running out
            if time.time() > deadline:
                return False
        
        return found
    
    # Try k = 1, 2, 3, ...
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Try to find a cover with exactly k sets
        # We need to be smarter: use branch-and-bound that tries to use exactly k sets
        # But for simplicity, let's use the DP result as an upper bound
        
        # Actually, let's just use the DP approach but with better pruning
        # For now, let's compute DP to get the answer
        pass
    
    # Fall back to DP if branch-and-bound is too complex
    INF = n + 1
    dp = [INF] * N
    parent = [0] * N
    dp[0] = 0
    
    # Precompute valid submasks for each S
    # For efficiency, precompute the list of valid subsets for each S
    valid_submasks = [[] for _ in range(N)]
    for S in range(1, N):
        T = S
        while T:
            if valid[T]:
                valid_submasks[S].append(T)
            T = (T - 1) & S
    
    for S in range(1, N):
        if time.time() > deadline:
            # Return a valid partition: each vertex as its own part
            fallback = [[i] for i in range(n)]
            return [list(map(int, p)) for p in fallback]
        
        best = INF
        bestT = 0
        for T in valid_submasks[S]:
            prev = S ^ T
            if dp[prev] + 1 < best:
                best = dp[prev] + 1
                bestT = T
        dp[S] = best
        parent[S] = bestT
    
    # Reconstruct
    S = full
    parts = []
    while S:
        T = parent[S]
        if T == 0:
            # fallback: single-vertex parts for remaining
            for i in range(n):
                if (S >> i) & 1:
                    parts.append([i])
            S = 0
            break
        vertices = [i for i in range(n) if (T >> i) & 1]
        parts.append(vertices)
        S ^= T
    
    # Convert to plain Python ints/lists
    result = [list(map(int, p)) for p in parts]
    return result
# EVOLVE-BLOCK-END
