"""Rewritten program for matrix multiplication 2x3 by 3x4.
Uses a vectorized construction approach to define the factor matrices U, V, W.
The rank is the product of dimensions n*m*p = 24.
"""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank of the tensor for standard matrix multiplication is n * m * p
    rank = n * m * p
    
    # Initialize factor matrices as zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Construct the decomposition
    # For each triple (i, j, k) where 0<=i<n, 0<=j<m, 0<=k<p:
    # The term is A[i,j] * B[j,k] contributing to C[i,k]
    # In the tensor decomposition C = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # Specifically, C[i,k] = sum_r U[i*m+j? No, U is indexed by input A, V by input B, W by output C?
    # Let's check the convention in the seed:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # So U corresponds to A indices (i,j)
    # V corresponds to B indices (j,k)
    # W corresponds to C indices (i,k) but mapped to linear index k*n+i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                
                # U index: i*m + j
                U[i * m + j, col] = 1
                
                # V index: j*p + k
                V[j * p + k, col] = 1
                
                # W index: k*n + i
                W[k * n + i, col] = 1

    # Convert numpy arrays to lists of lists to match the interface expected by the evaluator
    # if it requires plain python types, though numpy arrays are often accepted.
    # The seed used lists. I will provide lists to be safe.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
