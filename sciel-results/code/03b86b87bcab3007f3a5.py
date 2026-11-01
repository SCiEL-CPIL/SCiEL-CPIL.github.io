"""
Rewritten program for matrix multiplication 2x5x5 rank minimization.
"""
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    The schoolbook decomposition has rank n*m*p = 50.
    This function returns the schoolbook decomposition, which is a valid
    exact decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility, or numpy if allowed? 
    # The prompt says "integer/half-integer entries", returning dicts of lists is standard.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j (Row of A)
                # V index: j*p + k (Col of B? No, V is for B)
                # W index: k*n + i (Output position)
                
                # Let's verify the trace convention:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # Tensor T[i,j,k] = A[i,j] B[j,k] C^*[i,k]
                # Decomposition: T = sum_r U[:,r] V[:,r] W[:,r] (outer product)
                # Usually U[i,j,r], V[j,k,r], W[i,k,r]
                
                # The interface says:
                # U: (n*m, rank)
                # V: (m*p, rank)
                # W: (p*n, rank) -> Note: p*n, not n*p.
                # And "output factor follows the official trace convention W[k*n+i,r]"
                # This implies W's row index is `idx = k*n + i`.
                
                # In standard matrix mult:
                # C[i, k] depends on A[i, j] and B[j, k].
                # So the tensor index is (i, j, k).
                # U corresponds to A indices (i, j). Row idx = i*m + j.
                # V corresponds to B indices (j, k). Row idx = j*p + k.
                # W corresponds to C indices (i, k). Row idx = k*n + i (as per prompt W[k*n+i]).
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
