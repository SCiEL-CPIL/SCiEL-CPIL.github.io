"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    # Try to find a lower rank decomposition
    # The trivial rank is 63. Let's try to find the minimal rank.
    # For 3x3 * 3x7, lower bound is max(9, 21, 21) = 21
    
    # Try rank 21 decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as sum over r of terms
    # For rank 21, use one term per (j,k) pair
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for j in range(m):
        for k in range(p):
            # Term r: U picks out A[i,j] for all i, V picks out B[j,k], W distributes to C[i,k]
            for i in range(n):
                U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            for i in range(n):
                W[k * n + i, col] = 1
            col += 1
    
    # Verify: C[i,k] should equal sum_j A[i,j]*B[j,k]
    # For our decomposition: C[i,k] = sum_r W[k*n+i,r] * (sum_{i',j'} U[i'*m+j',r]*A[i',j']) * (sum_{j',k'} V[j'*p+k',r]*B[j',k'])
    # For term r corresponding to (j0, k0):
    #   sum_{i',j'} U[i'*m+j',r]*A[i',j'] = sum_i A[i,j0] (since U[i*m+j0,r]=1 for all i, 0 otherwise)
    #   Wait, that's sum_i A[i,j0], not A[i,j0]. That's wrong.
    
    # The issue: U[i*m+j, r] should pick out A[i,j] specifically, not sum over i.
    # So for each (j,k) pair, we need separate terms for each i.
    # That gives back rank 63.
    
    # Let me try a different approach. Use the standard algorithm structure.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is inherently rank 63 for the trivial decomposition.
    # 
    # To reduce rank, we need to exploit linearity.
    # 
    # Actually, let me just return the trivial decomposition for now,
    # but try to see if a smaller rank works by construction.
    
    # For now, use the trivial rank-63 decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
