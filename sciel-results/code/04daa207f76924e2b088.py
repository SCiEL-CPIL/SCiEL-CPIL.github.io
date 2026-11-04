"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    full_rank = nm * mp * pn  # 3*5*6 = 90, but full tensor has nm*mp*pn entries? No, tensor is (n*m, m*p, p*n) with size 15*30*18=8100
    
    # The tensor T[i,j,k] = 1 if j is the middle index connecting A[i,j] and B[j,k] to C[i,k]
    # Actually the matrix multiplication tensor: T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has dimensions (n, m, p) and T[i,j,k] = 1 for all valid i,j,k.
    # But the factor matrices have shapes (n*m, rank), (m*p, rank), (p*n, rank).
    # U is indexed by (i,j) -> i*m+j
    # V is indexed by (j,k) -> j*p+k
    # W is indexed by (k,i) -> k*n+i (trace convention: W[k*n+i, r])
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Trivial: rank = n*m*p, with U[i*m+j, i*m*p + j*p + k] = 1, V[j*p+k, i*m*p + j*p + k] = 1, W[k*n+i, i*m*p + j*p + k] = 1
    # This gives rank = n*m*p = 90.
    
    # We want to find a lower rank decomposition.
    # For small matrices, known results:
    # 2x2x2: rank 7 (Strassen)
    # 2x3x3, 3x2x3, 3x3x2: rank 18 or less
    # 2x2x3: rank 13
    # 2x3x2: rank 13
    # 3x2x2: rank 13
    # 3x3x3: rank 23 (or 22?)
    # 2x2x4: rank 17
    # 2x4x2: rank 17
    # 4x2x2: rank 17
    # 2x3x4, 3x2x4, 4x2x3, 2x4x3, 3x4x2, 4x3x2: various
    # 3x5x6: unknown, but we can try to find a decomposition.
    
    # Strategy: Use a rank-r decomposition with rational entries.
    # We can try to find a decomposition using a greedy or algebraic approach.
    
    # For now, let's try to find a decomposition with rank < 90.
    # We'll use a simple approach: try to find a decomposition with rank r by solving a system of equations.
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # We want U, V, W such that for all (a, b, c) in [0,nm)x[0,mp)x[0,pn):
    #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is a system of nm*mp*pn equations in nm*r + mp*r + pn*r unknowns.
    # For r = 90, this is underdetermined and the trivial solution works.
    # For r < 90, we need to find a solution.
    
    # Let's try a random approach: generate random U, V, W with small integer entries and check if they work.
    # This is unlikely to work for rank < 90, but let's try.
    
    # Actually, let's try a more systematic approach.
    # We can use the fact that the tensor is symmetric in some sense and try to find a decomposition.
    
    # For now, let's just return the trivial decomposition with rank 90.
    # We can try to improve this in future iterations.
    
    # Let's try to find a decomposition with rank 70, 60, 50, etc.
    # We'll use a simple greedy approach: start with the trivial decomposition and try to merge terms.
    
    # Actually, let's try a different approach: use the known decomposition for 2x2x2 (Strassen) and extend it.
    # But 3x5x6 is not a simple extension.
    
    # Let's try to find a decomposition with rank 80, 70, 60, 50, 40, 30, 20.
    # We'll use a random search with small integer entries.
    
    # For now, let's just return the trivial decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
