import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 50.0 # Safety margin
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full = (1 << n) - 1
    N = 1 << n
    
    # Find all valid subsets (cliques or independent sets)
    # We can generate them by checking all subsets, or more efficiently.
    # For n=14, 2^14 = 16384 subsets. Checking each is fast.
    
    valid_subsets = []
    
    # Precompute clique/indep status for all subsets
    # is_clique[S], is_indep[S]
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
        if is_clique[S] or is_indep[S]:
            valid_subsets.append(S)
            
    if not valid_subsets:
        # Fallback: single vertices
        return [[i] for i in range(n)]
        
    m = len(valid_subsets)
    
    # Setup MILP
    # Variables y_i for i in range(m)
    # Minimize sum(y_i)
    c = np.ones(m)
    
    # Constraints: For each vertex v, sum_{i: v in S_i} y_i = 1
    A_eq = np.zeros((n, m))
    for i, S in enumerate(valid_subsets):
        for v in range(n):
            if (S >> v) & 1:
                A_eq[v, i] = 1
                
    b_eq = np.ones(n)
    
    integrality = np.ones(m)
    bounds = Bounds(0, 1)
    
    constraints = LinearConstraint(A_eq, b_eq, b_eq)
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.success:
            # Extract solution
            parts = []
            for i in range(m):
                if res.x[i] > 0.5: # Binary variable
                    S = valid_subsets[i]
                    vertices = [v for v in range(n) if (S >> v) & 1]
                    parts.append(vertices)
            # Verify partition covers all vertices exactly once
            # (MILP guarantees this if solved correctly)
            return [list(map(int, p)) for p in parts]
        else:
            # Fallback to DP if MILP fails
            pass
    except Exception:
        pass
        
    # Fallback DP if MILP fails or not available
    INF = n + 1
    dp = [INF] * N
    parent = [0] * N
    dp[0] = 0
    
    # Use the precomputed valid_subsets for faster DP?
    # Or just iterate submasks.
    # Since we already have is_clique/is_indep, we can use them.
    
    for S in range(1, N):
        if time.time() > deadline:
             # Fallback: singletons
             return [[i] for i in range(n)]
        T = S
        best = INF
        bestT = 0
        while T:
            if is_clique[T] or is_indep[T]:
                prev = S ^ T
                if dp[prev] + 1 < best:
                    best = dp[prev] + 1
                    bestT = T
            T = (T - 1) & S
        dp[S] = best
        parent[S] = bestT
        
    S = full
    parts = []
    while S:
        T = parent[S]
        if T == 0:
            for i in range(n):
                if (S >> i) & 1:
                    parts.append([i])
            S = 0
            break
        vertices = [i for i in range(n) if (T >> i) & 1]
        parts.append(vertices)
        S ^= T
        
    return [list(map(int, p)) for p in parts]
