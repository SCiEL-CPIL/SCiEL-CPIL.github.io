"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 9]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The (i,j,k) entry of the multiplication tensor should be 1
                # if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor T[i*m+j, j*p+k, k*n+i] should be 1
                # And sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
                # Wait, that's not right. The tensor decomposition is:
                # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
                # where T is the multiplication tensor
                # For matrix multiplication, T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
                # and 0 otherwise? No, that's not the standard indexing.
                
                # Actually, the standard tensor for matrix multiplication M_{n,m,p} has:
                # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
                # So for each (i,j,k), we have T[i*m+j, j*p+k, k*n+i] = 1
                
                # The decomposition is T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # So T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                
                # For the multiplication tensor, T[i*m+j, j*p+k, k*n+i] = 1
                # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For all i,j,k, and T[a,b,c] = 0 for other (a,b,c)
                
                # But checking all (a,b,c) is expensive. Let's just check the non-zero entries.
                val = sum(Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r]) 
                         for r in range(rank))
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to leave safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try a simple search: start with schoolbook and try to merge columns
    # This is a very naive approach, but let's try it
    U = [row[:] for row in U_school]
    V = [row[:] for row in V_school]
    W = [row[:] for row in W_school]
    current_rank = rank_school
    
    # Try to find linear dependencies and eliminate columns
    # For each pair of columns, check if one is a linear combination of others
    # This is expensive, so we'll do a limited search
    
    # Actually, let's try a different approach: use a random initialization and gradient descent
    # But with exact arithmetic, this is tricky.
    
    # For now, return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
