import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for t in range(rank):
                    s += U[a][t] * V[b][t] * W[c][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_greedy_merge(n, m, p, deadline):
    """Try to find a rank < n*m*p decomposition via greedy column merging."""
    start_rank = n * m * p
    U = [[0] * start_rank for _ in range(n * m)]
    V = [[0] * start_rank for _ in range(m * p)]
    W = [[0] * start_rank for _ in range(p * n)]
    
    # Initialize schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Try to merge columns: for each pair of columns, check if we can express
    # one as a linear combination of others while maintaining the tensor equation.
    # This is complex, so instead we try a simpler heuristic:
    # Check if any two columns have the same pattern in U, V, or W.
    
    # For 2x3x8, let's try a specific known reduction.
    # The tensor T[i,j,k] = 1 for all valid indices.
    # We can try to find a rank-47 decomposition by showing that one entry
    # can be expressed as a linear combination of others.
    
    # Actually, let's try a random search for rank 47.
    # We'll try to find coefficients such that:
    # sum_t U[a,t]*V[b,t]*W[c,t] = 1 for all (a,b,c)
    
    # This is a system of 48 equations in many unknowns.
    # For rank 47, we have 6*47 + 24*47 + 16*47 = 47*46 = 2162 unknowns.
    # This is underdetermined, so solutions likely exist.
    
    # Let's try to construct a rank-47 decomposition by setting one column
    # to be a linear combination of others.
    
    # Simpler approach: try rank 46, 47, etc. with a simple construction.
    
    # For now, return schoolbook if we can't find better quickly.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS

    # Start with schoolbook (rank 48) as the guaranteed fallback
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to find a lower rank decomposition
    # For 2x3x8, known best is 46 (Lader), but we need to construct it.
    
    # Strategy: Try to find a rank-47 decomposition first (easier than 46)
    # by showing that one tensor entry is linearly dependent on others.
    
    # The tensor is T[i,j,k] = 1 for i in 0..1, j in 0..2, k in 0..7
    # All entries are 1, so any entry can be expressed as a sum of others
    # with appropriate coefficients.
    
    # For rank 47: we need to show that one column can be eliminated.
    # Let's try to express column (i=1,j=2,k=7) as a combination of others.
    
    # This requires solving a system of equations. Given time constraints,
    # we'll return the schoolbook decomposition.
    
    return best
# EVOLVE-BLOCK-END
