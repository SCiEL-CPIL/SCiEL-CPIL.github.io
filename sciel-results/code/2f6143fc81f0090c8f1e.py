"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor: T[i,j,k] = 1 if (i,j) in A, (j,k) in B
    # The decomposition says: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # We check this by building the sum for each (i,j,k).
    # For efficiency, we can check: for each c, the outer product structure.
    # Actually, the tensor is T_{ijk} = delta_{j,j} (which is always 1 for valid j).
    # Wait, matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k in valid ranges.
    # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Let's verify by checking all (i,j,k) tuples.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s kill
    
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But actually the tensor rank of matrix multiplication is at most n*m*p and at least max(n*m, m*p, p*n).
    # For 2x5 * 5x5, schoolbook gives rank 50.
    # Let's try to find a better decomposition.
    
    # Approach: Use the fact that the tensor is separable in a certain way.
    # T[i,j,k] = 1 = 1 * 1 * 1
    # This is already rank 1 if we could use constant vectors, but the indices are tied.
    
    # Actually, let's think about this more carefully.
    # The matrix multiplication tensor for C = A*B where A is n x m and B is m x p:
    # T_{i,j,k} = 1 (the coefficient of A_{i,j} * B_{j,k} in C_{i,k})
    
    # One approach: decompose using the structure of the indices.
    # For each j, the "slice" T[:,j,:] is a n x p all-ones matrix.
    # An all-ones matrix of size n x p has tensor rank 1 (it's the outer product of ones vectors).
    
    # So we can write: T = sum_j (ones_n ⊗ e_j ⊗ ones_p) where e_j is the j-th standard basis vector in R^m
    # This gives rank m = 5.
    
    # Let's verify: for this decomposition,
    # U[i*m+j, c] where c corresponds to j': U[i*m+j, j'] = 1 if j=j' else 0... 
    # Wait, that's not quite right. Let me re-derive.
    
    # If we use rank m decomposition:
    # For each j in 0..m-1, define:
    #   U_c[i*m+j'] = 1 if j'=j else 0  (for all i)
    #   V_c[j'*p+k] = 1 if j'=j else 0  (for all k)  
    #   W_c[k'*n+i'] = 1 for all k',i'
    
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = sum_j U[i*m+j,j]*V[j*p+k,j]*W[k*n+i,j]
    # = sum_j [1 if j=j else 0] * [1 if j=j else 0] * 1
    # Wait, U[i*m+j,c] with c=j' means: U[i*m+j, j'] = 1 if j'==j
    # V[j*p+k,c] with c=j' means: V[j*p+k, j'] = 1 if j'==j
    # So the product is 1 only when j'=j, and then it's 1*1*1 = 1.
    # Sum over j' gives 1. 
    
    # So rank m = 5 works! Let me construct it.
    
    rank = m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        # U: for each i, U[i*m+j, j] = 1
        for i in range(n):
            U[i * m + j][j] = 1
        # V: for each k, V[j*p+k, j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        # W: W[k*n+i, j] = 1 for all k,i
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification failed (shouldn't happen), fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
