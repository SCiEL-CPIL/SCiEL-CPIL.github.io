"""Cochromatic partition via subset DP."""
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
    
    # DP: dp[S] = min number of valid parts to partition S
    INF = n + 1
    dp = [INF] * N
    parent = [0] * N
    dp[0] = 0
    
    # Precompute list of valid submasks for each S to speed up DP
    # For each S, we want to iterate only over valid T ⊆ S
    # But that's expensive to precompute. Instead, we'll just iterate submasks.
    
    for S in range(1, N):
        if time.time() > deadline:
            # Return a valid partition: each vertex as its own part
            fallback = [[i] for i in range(n)]
            return [list(map(int, p)) for p in fallback]
        
        T = S
        best = INF
        bestT = 0
        while T:
            if valid[T]:
                prev = S ^ T
                if dp[prev] + 1 < best:
                    best = dp[prev] + 1
                    bestT = T
            T = (T - 1) & S
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
