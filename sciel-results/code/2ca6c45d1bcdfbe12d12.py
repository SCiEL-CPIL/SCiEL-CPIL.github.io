"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin before 120s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using structured patterns
    # Strategy: Try to combine multiple schoolbook terms into fewer terms
    # by exploiting the structure of the matrix multiplication tensor
    
    # For 2x4 * 4x6, the tensor has entries T[i,j,k] where:
    # - i in [0,2) (rows of A)
    # - j in [0,4) (cols of A / rows of B)
    # - k in [0,6) (cols of B)
    # The output C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-1 term is defined by:
    # - U indices: (i, j) for some i, j
    # - V indices: (j, k) for some j, k
    # - W indices: (k, i) for some k, i
    # The constraint is that the j index must match between U and V
    
    # Let's try to find a decomposition with rank < 48
    # We'll use a greedy approach: try to cover all (i,j,k) triples with
    # rank-1 terms where each term covers multiple triples
    
    # First, let's try a structured approach based on grouping
    # Group by j index: for each j, we have n*p = 12 (i,k) pairs
    # But we can potentially share terms across different j values
    
    # Try rank values from 24 to 47
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a decomposition with rank around 32-40
    # using a pattern-based approach
    
    # Pattern 1: Group by pairs of j indices
    # For each pair (j1, j2), create terms that cover both j1 and j2
    
    # Actually, let's try a different approach: use the fact that
    # for fixed i and k, we need sum over j of A[i,j]*B[j,k]
    # This is a bilinear form in j
    
    # Let's try to find a low-rank decomposition by solving a system
    # of equations. For rank r, we need to find U (n*m x r), V (m*p x r),
    # W (p*n x r) such that for all i,j,k:
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # This is a 3-tensor decomposition problem. Let's try a simple
    # structured approach first.
    
    # Try: for each i in [0,n), for each k in [0,p), we need to represent
    # the vector (A[i,0]*B[0,k], A[i,1]*B[1,k], ..., A[i,m-1]*B[m-1,k])
    # as a sum of rank-1 terms
    
    # Actually, let's just try to verify if rank 48 is optimal by checking
    # a few candidate ranks with a simple search
    
    # For now, return the schoolbook decomposition
    # A proper search would require more sophisticated algorithms
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
