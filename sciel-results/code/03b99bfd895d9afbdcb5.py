"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110  # safety margin before 120s limit

    # Build the tensor T[i,j,k] = 1 for all (i,j,k) in range
    # Rank-r decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Schoolbook: one term per (i,j,k), rank = n*m*p = 120

    # Represent each rank-1 term as a triple of vectors
    # Term (i,j,k): U has 1 at pos i*m+j, V has 1 at pos j*p+k, W has 1 at pos k*n+i
    # We want to find a subset of terms that still spans the tensor

    # Key insight: the tensor is determined by its values on all (i,j,k)
    # Two decompositions are equivalent if they give same T[i,j,k] for all i,j,k
    # We can try to eliminate a term if it's a linear combination of others

    # For efficiency, we'll use a greedy approach:
    # 1. Start with all 120 terms
    # 2. Try to eliminate terms one by one
    # 3. Check if the remaining terms still span the tensor

    # To check spanning, we need to verify that for each (i,j,k),
    # there exists a coefficient assignment that reproduces T[i,j,k]=1

    # This is complex. Instead, let's use a different approach:
    # Use the fact that the tensor has a specific structure and try known constructions

    # For now, return schoolbook as fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Try a simple optimization: see if we can reduce rank by checking
    # if any term is redundant given the others
    # This requires solving a system of equations, which is expensive

    # Given time constraints, return schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
