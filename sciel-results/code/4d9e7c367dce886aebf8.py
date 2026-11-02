"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-15 decomposition exploiting the structure:
    # For C = A*B where A is n x m and B is m x p,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We use one term per (i,k) output entry.
    # Term c = i*p + k (for i in range(n), k in range(p)):
    #   U[i*m+j, c] = 1 if row index matches i (i.e., (i*m+j)//m == i)
    #   V[j*p+k, c] = 1 if col index matches k (i.e., (j*p+k)%p == k)
    #   W[k*n+i, c] = 1
    #
    # This gives: for fixed (i',k'), the sum over c and j of
    #   U[i'*m+j, c] * V[j*p+k', c] * W[k'*n+i', c]
    # Only c = i'*p + k' has W[k'*n+i', c] = 1, and for that c:
    #   U[i'*m+j, c] = 1 for all j (since row is i')
    #   V[j*p+k', c] = 1 for all j (since col is k')
    # So the sum over j gives m = 5... wait, that's not right.
    #
    # Actually, let me reconsider. The decomposition should satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # (so that C[i,k] = sum_j 1 * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k])
    #
    # With my construction: for fixed (i,j,k), which c values contribute?
    # W[k*n+i, c] = 1 only when c = i*p + k (since W is indexed by k*n+i)
    # Wait, W has shape (p*n, rank), indexed by [k*n+i].
    # So W[k*n+i, c] = 1 iff the output entry (i,k) corresponds to term c.
    # If c = i*p + k, then W[k*n+i, c] = 1.
    # But also need U[i*m+j, c] and V[j*p+k, c] to be non-zero.
    #
    # Let me redefine: for term c corresponding to output (i_c, k_c) where c = i_c*p + k_c:
    #   U[i*m+j, c] = 1 if i == i_c (i.e., the row of A matches)
    #   V[j*p+k, c] = 1 if k == k_c (i.e., the col of B matches)
    #   W[k'*n+i', c] = 1 if (i', k') == (i_c, k_c)
    #
    # Then for fixed (i,j,k), the sum over c:
    # Only c = i*p + k can have W[k*n+i, c] = 1 (since W[k'*n+i', c] = 1 iff i'=i_c and k'=k_c, so we need i'=i and k'=k, thus c = i*p+k)
    # For c = i*p+k: U[i*m+j, c] = 1 (since i == i_c = i), V[j*p+k, c] = 1 (since k == k_c = k)
    # So the sum is 1. 
    #
    # For other c values, W[k*n+i, c] = 0, so they don't contribute.
    #
    # This works! Rank = n*p = 15.
    
    rank = n * p  # 15
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            # U[i*m+j, c] = 1 for all j (row i of A)
            for j in range(m):
                U[i * m + j][c] = 1
            # V[j*p+k, c] = 1 for all j (col k of B)
            for j in range(m):
                V[j * p + k][c] = 1
            # W[k*n+i, c] = 1
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
