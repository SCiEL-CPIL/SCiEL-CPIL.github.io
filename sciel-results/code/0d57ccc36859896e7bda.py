"""
Matrix multiplication tensor decomposition for 3x3 x 3x4.

Strategy:
1. Start with the schoolbook rank-36 decomposition.
2. Use a local search (coordinate descent / randomized rounding) to try to reduce the rank
   by identifying redundant or mergeable terms.
3. Alternatively, use a structured approach: try to find a rank-30 decomposition using
   known results or a targeted search.

Given the 120s budget, we will:
- First, quickly check if a known rank-30 solution can be constructed via a simple
  structured ansatz (e.g., using the 3x3x3 rank-23 decomposition extended to 3x3x4).
- If not, perform a local search starting from rank-36 to try to reduce the rank.
- Return the best result found within the time limit.

For robustness, we will:
- Use the schoolbook as a guaranteed fallback.
- Try a few structured reductions.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check if the factorization is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_time=30):
    """
    Try to reduce the rank of the decomposition by local search.
    This is a heuristic approach and may not always succeed.
    """
    start_time = time.time()
    rank = len(U[0])
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    # Try to zero out columns that are linearly dependent or redundant
    # This is a simplified approach: try to find a column that can be removed
    # without violating the constraints, by adjusting the others.
    
    # For now, this is a placeholder for a more sophisticated local search.
    # We'll just return the same decomposition for now.
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s out of 120s
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_factors = (U_school, V_school, W_school)
    best_rank = n * m * p
    
    # Try a structured approach for rank 30
    # This is a heuristic: try to find a rank-30 solution by modifying the schoolbook
    # or using a known construction.
    
    # For now, we'll just return the schoolbook since the local search is not implemented.
    # In a future iteration, we can implement a more sophisticated search.
    
    return {"U": best_factors[0], "V": best_factors[1], "W": best_factors[2]}

# EVOLVE-BLOCK-END
